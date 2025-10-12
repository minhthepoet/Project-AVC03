# train.py
# SwiftEdit unified training using your repo (models.py, src/*) + your losses.py
# Stage-1: train F_theta + IP-Adapter; Stage-2: train F_theta only with paper-correct regu loss.

import argparse, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize, to_pil_image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from models import InverseModel, IPSBV2Model, AuxiliaryModel, tokenize_captions

# ---- try to import your losses.py (preferred) ----
USE_EXT = True
try:
    from losses import (
        PerceptualLoss as ExtPerceptualLoss,
        stage1_loss as ext_stage1_loss,
        stage2_regu_loss as ext_stage2_regu_loss,  # paper-correct L_regu: E[w(t)||eps_phi(z_t,t,cy)-eps_hat||^2]
    )
except Exception:
    USE_EXT = False

# -----------------------------
# Dataset
# -----------------------------
class ImagePromptDataset(Dataset):
    def __init__(self, images_dir: str, prompts_file: str, image_size: int = 512):
        self.paths = []
        p = Path(images_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            self.paths += list(p.glob(ext))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {images_dir}")

        with open(prompts_file, "r", encoding="utf-8") as f:
            self.prompts = [ln.strip() for ln in f if ln.strip()]
        if len(self.prompts) == 0:
            raise RuntimeError(f"No prompts found in {prompts_file}")

        self.image_size = image_size

    def __len__(self):
        return max(len(self.paths), len(self.prompts))

    def __getitem__(self, idx):
        img_path  = self.paths[idx % len(self.paths)]
        prompt    = self.prompts[idx % len(self.prompts)]
        img = read_image(str(img_path)).float() / 255.0
        if img.shape[0] == 1:   img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4: img = img[:3]
        img = resize(img, [self.image_size, self.image_size],
                     interpolation=InterpolationMode.BICUBIC, antialias=True)
        return img, prompt


def collate_fn(batch):
    imgs, prompts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(prompts)


# -----------------------------
# Helpers
# -----------------------------
def requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


def get_scheduler(obj):
    # prefer scheduler from aux, else from ip_sb if available
    return getattr(obj, "scheduler", None)


def get_alpha_bar_terms(scheduler, t_tensor: torch.Tensor, device):
    # DDPM alpha_bar utilities for stage-2 regularization
    alphas_cumprod = scheduler.alphas_cumprod.to(device)  # [T]
    a_bar = alphas_cumprod[t_tensor]                      # [B]
    sqrt_ab = a_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus = (1.0 - a_bar).sqrt().view(-1, 1, 1, 1)
    return sqrt_ab, sqrt_one_minus


# -----------------------------
# Fallback losses (if your losses.py is missing pieces)
# -----------------------------
class FallbackPerceptual(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mode = "L1"
        try:
            from dists_pytorch import DISTS
            self.impl = DISTS().to(device)
            self.mode = "DISTS"
        except Exception:
            try:
                import lpips
                self.impl = lpips.LPIPS(net='vgg').to(device)
                self.mode = "LPIPS"
            except Exception:
                self.impl = nn.L1Loss()
        print(f"[Perceptual] Using {self.mode}")

    def forward(self, x, y):
        if self.mode == "DISTS":
            return self.impl(x, y)
        elif self.mode == "LPIPS":
            return self.impl(x*2-1, y*2-1).mean()
        else:
            return self.impl(x, y)


def fallback_stage1_loss(eps_hat, eps, z0_student, z0_target, lambda_regr=1.0):
    mse = nn.MSELoss()
    L_regr = mse(eps_hat, eps)
    L_rec  = mse(z0_student, z0_target)
    return L_rec + lambda_regr * L_regr, L_rec.detach(), L_regr.detach()


@torch.no_grad()
def _teacher_eps(teacher_unet, z_t, t, text_embeds):
    # SD 2.1 UNet predicts eps by default
    return teacher_unet(z_t, t, text_embeds).sample


def fallback_stage2_regu_loss(teacher_unet, scheduler, z_latent, eps_hat, text_embeds,
                              tmin=200, tmax=800, w_lambda=1.0, device="cuda"):
    """
    Paper-correct Stage-2 regularization:
      L_regu = E_t [ w(t) * || eps_phi(z_t, t, c_y) - eps_hat ||^2 ]
    where z_t = sqrt(a_bar) * z + sqrt(1-a_bar) * eps_rand and z is VAE latent of real image.
    """
    B = z_latent.size(0)
    t_rand = torch.randint(low=tmin, high=tmax + 1, size=(B,), device=device, dtype=torch.int64)
    sqrt_ab, sqrt_1m = get_alpha_bar_terms(scheduler, t_rand, device)
    eps_rand = torch.randn_like(z_latent)

    z_t = sqrt_ab * z_latent + sqrt_1m * eps_rand
    te_dtype = next(teacher_unet.parameters()).dtype
    teacher_pred = _teacher_eps(teacher_unet, z_t.to(te_dtype), t_rand, text_embeds.to(te_dtype))
    mse = nn.MSELoss()
    # weight w(t): use 1.0 (you can customize in losses.py if you want)
    L_regu = w_lambda * mse(teacher_pred.to(z_latent.dtype), eps_hat)
    return L_regu


# -----------------------------
# Stage-1
# -----------------------------
def run_stage1(args, device, inverse_model, aux_model, ip_sb_model, dl):
    # Train Fθ + IP-Adapter branch; freeze generator & encoders
    ip_sb_model.unet.requires_grad_(False)
    requires_grad(ip_sb_model.image_proj_model, True)
    for m in ip_sb_model.unet.attn_processors.values():
        requires_grad(m, True)

    requires_grad(inverse_model.vae, False)
    requires_grad(inverse_model.text_encoder, False)
    # image encoder is inside aux_model
    requires_grad(aux_model.image_encoder, False)

    inverse_model.unet_inverse.train()
    requires_grad(inverse_model.unet_inverse, True)

    # Optimizer
    params = list(inverse_model.unet_inverse.parameters()) \
           + list(ip_sb_model.image_proj_model.parameters())
    for m in ip_sb_model.unet.attn_processors.values():
        params += list(m.parameters())
    optimizer = optim.AdamW(params, lr=args.stage1_lr,
                            weight_decay=args.stage1_weight_decay, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # constants
    alpha_t = ip_sb_model.alpha_t
    sigma_t = ip_sb_model.sigma_t
    mid_timestep = torch.ones((1,), dtype=torch.int64, device=device) * args.stage1_mid_t

    os.makedirs(args.stage1_out, exist_ok=True)
    global_step = 0

    for epoch in range(args.stage1_epochs):
        pbar = tqdm(dl, desc=f"[Stage-1] Epoch {epoch+1}/{args.stage1_epochs}", leave=False)
        for imgs, prompts in pbar:
            imgs = imgs.to(device)

            with torch.no_grad():
                # encode images (use inverse_model.vae to match infer.py)
                latents = inverse_model.vae.encode(imgs.to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * inverse_model.vae.config.scaling_factor  # [B,4,h,w]

                # text embeds from inverse_model encoders (match infer.py)
                input_ids = tokenize_captions(inverse_model.tokenizer, prompts).to(device)
                text_embeds = inverse_model.text_encoder(input_ids)[0].to(dtype=torch.float32)

                # image embeds via aux_model -> IP projection
                pil_images = [to_pil_image(img.cpu()) for img in imgs]
                clip_image = aux_model.clip_image_processor(images=pil_images, return_tensors="pt").pixel_values
                clip_image_embeds = aux_model.image_encoder(clip_image.to(device, dtype=torch.float32)).image_embeds
                image_prompt_embeds = ip_sb_model.image_proj_model(clip_image_embeds)
                prompt_embeds = torch.cat([text_embeds, image_prompt_embeds], dim=1)

            # sample ground-truth noise
            eps = torch.randn_like(latents, device=device)

            # teacher path (frozen generator)
            with torch.no_grad():
                noise_teacher = alpha_t * latents + sigma_t * eps
                model_pred_teacher = ip_sb_model.unet(noise_teacher, ip_sb_model.timestep, prompt_embeds).sample
                if model_pred_teacher.shape[1] == noise_teacher.shape[1] * 2:
                    model_pred_teacher, _ = torch.split(model_pred_teacher, noise_teacher.shape[1], dim=1)
                z0_target = (noise_teacher - sigma_t * model_pred_teacher) / alpha_t

            with torch.cuda.amp.autocast(enabled=args.fp16):
                eps_hat = inverse_model.unet_inverse(latents, mid_timestep.expand(latents.size(0)),
                                                     text_embeds).sample
                noise_student = alpha_t * latents + sigma_t * eps_hat
                model_pred_student = ip_sb_model.unet(noise_student, ip_sb_model.timestep, prompt_embeds).sample
                if model_pred_student.shape[1] == noise_student.shape[1] * 2:
                    model_pred_student, _ = torch.split(model_pred_student, noise_student.shape[1], dim=1)
                z0_student = (noise_student - sigma_t * model_pred_student) / alpha_t

                if USE_EXT and 'ext_stage1_loss' in globals():
                    loss, L_rec, L_regr = ext_stage1_loss(eps_hat, eps, z0_student, z0_target,
                                                          lambda_regr=args.stage1_lambda_regr)
                else:
                    loss, L_rec, L_regr = fallback_stage1_loss(eps_hat, eps, z0_student, z0_target,
                                                               lambda_regr=args.stage1_lambda_regr)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            pbar.set_postfix({"L": float(loss.detach().cpu()),
                              "L_rec": float(L_rec.detach().cpu()),
                              "L_regr": float(L_regr.detach().cpu())})

            if global_step % args.stage1_save_every == 0:
                save_path = os.path.join(args.stage1_out, f"swiftedit_stage1_step{global_step}.pt")
                torch.save({
                    "step": global_step,
                    "unet_inverse": inverse_model.unet_inverse.state_dict(),
                    "image_proj_model": ip_sb_model.image_proj_model.state_dict(),
                    "adapter_modules": {n: m.state_dict() for n, m in ip_sb_model.unet.attn_processors.items()},
                }, save_path)

            if global_step >= args.stage1_max_steps:
                break
        if global_step >= args.stage1_max_steps:
            break

    final_path = os.path.join(args.stage1_out, "swiftedit_stage1_final.pt")
    torch.save({
        "step": global_step,
        "unet_inverse": inverse_model.unet_inverse.state_dict(),
        "image_proj_model": ip_sb_model.image_proj_model.state_dict(),
        "adapter_modules": {n: m.state_dict() for n, m in ip_sb_model.unet.attn_processors.items()},
    }, final_path)
    print(f"[Stage-1] Done. Saved to {final_path}")
    return final_path


# -----------------------------
# Stage-2
# -----------------------------
def run_stage2(args, device, inverse_model, aux_model, ip_sb_model, dl):
    # freeze generator + IP-Adapter
    ip_sb_model.unet.eval()
    requires_grad(ip_sb_model.unet, False)
    requires_grad(ip_sb_model.image_proj_model, False)
    for m in ip_sb_model.unet.attn_processors.values():
        requires_grad(m, False)

    # train F_theta only
    inverse_model.unet_inverse.train()
    requires_grad(inverse_model.unet_inverse, True)

    # teacher UNet (SD 2.1) for regu
    from diffusers import UNNet2DConditionModel as _maybe_typo  # guard against wrong import
    try:
        from diffusers import UNet2DConditionModel
        TeacherUNet = UNet2DConditionModel
    except Exception:
        # if diffusers alias changed, fallback to loaded in models (unlikely)
        from diffusers import UNet2DConditionModel as TeacherUNet

    teacher_unet = TeacherUNet.from_pretrained(args.base_model, subfolder="unet").to(
        device=device, dtype=torch.float16 if args.fp16 else torch.float32
    )
    teacher_unet.eval()
    requires_grad(teacher_unet, False)

    # losses
    if USE_EXT and 'ExtPerceptualLoss' in globals():
        perc = ExtPerceptualLoss(device)
    else:
        perc = FallbackPerceptual(device)

    optimizer = optim.AdamW(inverse_model.unet_inverse.parameters(),
                            lr=args.stage2_lr, weight_decay=args.stage2_weight_decay, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # constants
    alpha_t = ip_sb_model.alpha_t
    sigma_t = ip_sb_model.sigma_t
    scheduler = get_scheduler(aux_model) or get_scheduler(ip_sb_model)
    if scheduler is None:
        raise RuntimeError("No scheduler found on aux_model or ip_sb_model for stage-2 regularization.")

    os.makedirs(args.stage2_out, exist_ok=True)
    global_step = 0

    for epoch in range(args.stage2_epochs):
        pbar = tqdm(dl, desc=f"[Stage-2] Epoch {epoch+1}/{args.stage2_epochs}", leave=False)
        for imgs, prompts in pbar:
            imgs = imgs.to(device)

            with torch.no_grad():
                # encode real image latent z (constant wrt theta)
                z = inverse_model.vae.encode(imgs.to(dtype=torch.float32)).latent_dist.sample()
                z = z * inverse_model.vae.config.scaling_factor

                # embeddings
                input_ids = tokenize_captions(inverse_model.tokenizer, prompts).to(device)
                text_embeds = inverse_model.text_encoder(input_ids)[0].to(dtype=torch.float32)

                pil_images = [to_pil_image(img.cpu()) for img in imgs]
                clip_image = aux_model.clip_image_processor(images=pil_images, return_tensors="pt").pixel_values
                clip_image_embeds = aux_model.image_encoder(clip_image.to(device, dtype=torch.float32)).image_embeds
                image_prompt_embeds = ip_sb_model.image_proj_model(clip_image_embeds)
                prompt_embeds = torch.cat([text_embeds, image_prompt_embeds], dim=1)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                mid_t = torch.ones((imgs.size(0),), dtype=torch.int64, device=device) * args.stage2_mid_t
                eps_hat = inverse_model.unet_inverse(z, mid_t, text_embeds).sample  # \hat eps = F_theta(z,cy)

                # reconstruct via frozen generator (for perceptual)
                noise_student = alpha_t * z + sigma_t * eps_hat
                model_pred_student = ip_sb_model.unet(noise_student, ip_sb_model.timestep, prompt_embeds).sample
                if model_pred_student.shape[1] == noise_student.shape[1] * 2:
                    model_pred_student, _ = torch.split(model_pred_student, noise_student.shape[1], dim=1)
                z0_student = (noise_student - sigma_t * model_pred_student) / alpha_t

                latents_dec = z0_student / inverse_model.vae.config.scaling_factor
                x_hat = inverse_model.vae.decode(latents_dec).sample
                x_hat = (x_hat.clamp(-1, 1) + 1) / 2.0

                L_perc = perc(x_hat, imgs) * args.stage2_lambda_perc

                # paper-correct regu loss:
                if USE_EXT and 'ext_stage2_regu_loss' in globals():
                    L_regu = ext_stage2_regu_loss(
                        teacher_unet=teacher_unet,
                        scheduler=scheduler,
                        z_latent=z,
                        eps_hat=eps_hat,
                        text_embeds=text_embeds,
                        tmin=args.stage2_tmin,
                        tmax=args.stage2_tmax,
                        w_lambda=args.stage2_lambda_regu,
                        device=device,
                    )
                else:
                    L_regu = fallback_stage2_regu_loss(
                        teacher_unet=teacher_unet,
                        scheduler=scheduler,
                        z_latent=z,
                        eps_hat=eps_hat,
                        text_embeds=text_embeds,
                        tmin=args.stage2_tmin,
                        tmax=args.stage2_tmax,
                        w_lambda=args.stage2_lambda_regu,
                        device=device,
                    )

                loss = L_perc + L_regu

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            pbar.set_postfix({"L": float(loss.detach().cpu()),
                              "L_perc": float(L_perc.detach().cpu()),
                              "L_regu": float(L_regu.detach().cpu())})

            if global_step % args.stage2_save_every == 0:
                save_path = os.path.join(args.stage2_out, f"swiftedit_stage2_step{global_step}.pt")
                torch.save({"step": global_step, "unet_inverse": inverse_model.unet_inverse.state_dict()}, save_path)

            if global_step >= args.stage2_max_steps:
                break
        if global_step >= args.stage2_max_steps:
            break

    final_path = os.path.join(args.stage2_out, "swiftedit_stage2_final.pt")
    torch.save({"step": global_step, "unet_inverse": inverse_model.unet_inverse.state_dict()}, final_path)
    print(f"[Stage-2] Done. Saved to {final_path}")
    return final_path


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="SwiftEdit Training (Stage-1 / Stage-2 / Both)")
    ap.add_argument("--stage", type=str, choices=["stage1", "stage2", "both"], required=True)

    # base / teacher
    ap.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-2-1-base")

    # weights & init (match infer.py signatures)
    ap.add_argument("--inverse_init", type=str, required=True, help="Folder of inverse_ckpt-XXX (contains unet_ema)")
    ap.add_argument("--sbv2_unet_path", type=str, required=True, help="swiftedit_weights/sbv2_0.5")
    ap.add_argument("--ip_ckpt_in", type=str, required=True, help="swiftedit_weights/ip_adapter_ckpt-90k/ip_adapter.bin")

    # data
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--prompts_file", type=str, required=True)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)

    # stage-1 hparams
    ap.add_argument("--stage1_epochs", type=int, default=1)
    ap.add_argument("--stage1_max_steps", type=int, default=10000)
    ap.add_argument("--stage1_save_every", type=int, default=1000)
    ap.add_argument("--stage1_lr", type=float, default=1e-5)
    ap.add_argument("--stage1_weight_decay", type=float, default=1e-4)
    ap.add_argument("--stage1_lambda_regr", type=float, default=1.0)
    ap.add_argument("--stage1_mid_t", type=int, default=500)
    ap.add_argument("--stage1_out", type=str, default="stage1_out")

    # stage-2 hparams
    ap.add_argument("--stage2_epochs", type=int, default=1)
    ap.add_argument("--stage2_max_steps", type=int, default=8000)
    ap.add_argument("--stage2_save_every", type=int, default=1000)
    ap.add_argument("--stage2_lr", type=float, default=1e-5)
    ap.add_argument("--stage2_weight_decay", type=float, default=1e-4)
    ap.add_argument("--stage2_lambda_perc", type=float, default=1.0)
    ap.add_argument("--stage2_lambda_regu", type=float, default=1.0)  # weight for L_regu per paper
    ap.add_argument("--stage2_mid_t", type=int, default=500)
    ap.add_argument("--stage2_tmin", type=int, default=200)
    ap.add_argument("--stage2_tmax", type=int, default=800)
    ap.add_argument("--stage2_out", type=str, default="stage2_out")

    ap.add_argument("--fp16", action="store_true")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # build models exactly like infer.py style:
    inverse_model = InverseModel(args.inverse_init)
    aux_model     = AuxiliaryModel()
    ip_sb_model   = IPSBV2Model(args.sbv2_unet_path, args.ip_ckpt_in, aux_model, with_ip_mask_controller=False).to(device)

    # dataloader
    ds = ImagePromptDataset(args.images_dir, args.prompts_file, image_size=args.resolution)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)

    # run
    stage1_path = None
    if args.stage in ["stage1", "both"]:
        stage1_path = run_stage1(args, device, inverse_model, aux_model, ip_sb_model, dl)

    if args.stage in ["stage2", "both"]:
        # if you saved stage-1 weights for Fθ as .pt, load them here (optional)
        # Example:
        # if stage1_path:
        #     ckpt = torch.load(stage1_path, map_location="cpu")
        #     if "unet_inverse" in ckpt:
        #         inverse_model.unet_inverse.load_state_dict(ckpt["unet_inverse"], strict=False)
        _ = run_stage2(args, device, inverse_model, aux_model, ip_sb_model, dl)


if __name__ == "__main__":
    main()
