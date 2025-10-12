# train_stage2.py
# SwiftEdit - Stage 2 (Real-image fine-tuning) training script
#
# Paper: SwiftEdit Sec. 4.1 (Stage-2): freeze IP-Adapter & generator, train only F_theta
# Loss = Perceptual (DISTS) + SDS-inspired regularization.
#
# Inputs:
#   - Stage-1 checkpoint (to init F_theta and keep IP-Adapter frozen as learned)
#   - A folder of real images + a prompts file (one prompt per image or cycled)
#
# Output:
#   - A checkpoint .pt with "unet_inverse" weights updated.

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

from models import AuxiliaryModel, IPSBV2Model, InverseModel, tokenize_captions

# -----------------------------
# Optional perceptual metrics
# -----------------------------
class PerceptualLoss(nn.Module):
    """DISTS if available, else LPIPS if available, else L1."""
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.impl = None

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
                self.mode = "L1"

        print(f"[Perceptual] Using {self.mode}")

    def forward(self, x, y):
        # x, y in [0,1], shape [B,3,H,W]
        if self.mode == "DISTS":
            return self.impl(x, y)
        elif self.mode == "LPIPS":
            # lpips expects [-1,1]
            return self.impl(x * 2 - 1, y * 2 - 1).mean()
        else:
            return self.impl(x, y)


# -----------------------------
# Dataset
# -----------------------------
class RealImagePromptDataset(Dataset):
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
        img = read_image(str(img_path)).float() / 255.0   # [C,H,W] in [0,1]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
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


def save_stage2_checkpoint(save_path: str, inverse_model: InverseModel, step: int):
    ckpt = {
        "step": step,
        "unet_inverse": inverse_model.unet_inverse.state_dict(),
    }
    torch.save(ckpt, save_path)


# -----------------------------
# Alpha-bar utilities (DDPM-style)
# -----------------------------
def get_alpha_bar_terms(scheduler, t_tensor: torch.Tensor, device):
    # scheduler.alphas_cumprod[t] ∈ [0..1], shape [T]
    # t_tensor: [B] int64 timesteps
    alphas_cumprod = scheduler.alphas_cumprod.to(device)  # [T]
    a_bar = alphas_cumprod[t_tensor]                      # [B]
    sqrt_ab = a_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus = (1.0 - a_bar).sqrt().view(-1, 1, 1, 1)
    return sqrt_ab, sqrt_one_minus


# -----------------------------
# Training loop
# -----------------------------
def train_stage2(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # 1) Aux: VAE, text encoder, image encoder, scheduler
    aux = AuxiliaryModel(
        model_name=args.base_model,            # "stabilityai/stable-diffusion-2-1-base"
        image_encoder_path=args.ip_adapter_repo,
        device=str(device),
    )

    # 2) SBv2 + IP-Adapter (FROZEN at Stage-2)
    ip_sb = IPSBV2Model(
        pretrained_model_name_path=args.sbv2_unet_path,
        ip_model_path=args.ip_ckpt_in,       # load trained IP-Adapter weights from Stage-1
        aux_model=aux,
        device=str(device),
        with_ip_mask_controller=False,
    ).to(device)

    ip_sb.unet.eval()
    requires_grad(ip_sb.unet, False)
    requires_grad(ip_sb.image_proj_model, False)
    for m in ip_sb.unet.attn_processors.values():
        requires_grad(m, False)

    # 3) Inversion network (load Stage-1 and train)
    inverse = InverseModel(
        pretrained_model_name_path=args.inverse_init,  # folder containing "unet_ema" or Stage-1 ckpt
        model_name=args.turbo_model,
        dtype="fp16" if args.fp16 else "fp32",
        device=str(device),
    )
    # If Stage-1 checkpoint path provided, load its unet_inverse weights
    if args.stage1_ckpt and os.path.isfile(args.stage1_ckpt):
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
        w = ckpt.get("unet_inverse", None)
        if w is not None:
            inverse.unet_inverse.load_state_dict(w, strict=False)
            print(f"[Stage2] Loaded Stage-1 inverse weights from {args.stage1_ckpt}")
    inverse.unet_inverse.train()
    requires_grad(inverse.unet_inverse, True)

    # 4) Teacher UNet for SDS-regularization (SD2.1)
    from diffusers import UNet2DConditionModel
    teacher_unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet"
    ).to(device=device, dtype=torch.float16 if args.fp16 else torch.float32)
    teacher_unet.eval()
    requires_grad(teacher_unet, False)

    # 5) Optimizer + scaler
    optimizer = optim.AdamW(
        inverse.unet_inverse.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # 6) Data + perceptual
    ds = RealImagePromptDataset(args.images_dir, args.prompts_file, image_size=args.resolution)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    perc = PerceptualLoss(device)

    # 7) Timings & constants
    alpha_t = ip_sb.alpha_t  # scalar for t=999 (final timestep for SBv2)
    sigma_t = ip_sb.sigma_t
    mse = nn.MSELoss()

    # 8) Train
    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Stage2 Epoch {epoch+1}/{args.epochs}", leave=False)
        for imgs, prompts in pbar:
            imgs = imgs.to(device)  # [B,3,H,W]

            # Encode image -> latents
            with torch.no_grad():
                latents = aux.vae.encode(imgs.to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * aux.vae.config.scaling_factor  # [B,4,H/8,W/8]

                # Text & image embeds (for G^IP; we keep IP branch frozen)
                input_ids = tokenize_captions(aux.tokenizer, prompts).to(device)
                text_embeds = aux.text_encoder(input_ids)[0].to(dtype=torch.float32)

                pil_images = [to_pil_image(img.detach().cpu()) for img in imgs]
                clip_image = aux.clip_image_processor(images=pil_images, return_tensors="pt").pixel_values
                clip_image_embeds = aux.image_encoder(clip_image.to(device, dtype=torch.float32)).image_embeds
                image_prompt_embeds = ip_sb.image_proj_model(clip_image_embeds)  # [B,extra,1024]
                prompt_embeds = torch.cat([text_embeds, image_prompt_embeds], dim=1)

            # Forward Fθ -> eps_hat, then reconstruct via frozen G^IP
            with torch.cuda.amp.autocast(enabled=args.fp16):
                mid_timestep = torch.ones((imgs.size(0),), dtype=torch.int64, device=device) * args.mid_t
                eps_hat = inverse.unet_inverse(latents, mid_timestep, text_embeds).sample  # [B,4,h,w]

                noise_student = alpha_t * latents + sigma_t * eps_hat
                model_pred_student = ip_sb.unet(noise_student, ip_sb.timestep, prompt_embeds).sample
                if model_pred_student.shape[1] == noise_student.shape[1] * 2:
                    model_pred_student, _ = torch.split(model_pred_student, noise_student.shape[1], dim=1)
                z0_student = (noise_student - sigma_t * model_pred_student) / alpha_t  # [B,4,h,w]

                # Decode to image in [0,1]
                latents_dec = z0_student / aux.vae.config.scaling_factor
                x_hat = aux.vae.decode(latents_dec).sample  # [B,3,H,W]
                x_hat = (x_hat.clamp(-1, 1) + 1) / 2.0

                # Perceptual loss
                L_perc = perc(x_hat, imgs)

                # SDS-inspired regularization:
                # Sample a random t' and encourage teacher epsilon(z_t') ~= eps_rand
                # Compose z_t' from z0_student with random noise
                B = imgs.size(0)
                t_rand = torch.randint(low=args.sds_tmin, high=args.sds_tmax+1, size=(B,), device=device, dtype=torch.int64)
                sqrt_ab, sqrt_1mab = get_alpha_bar_terms(aux.scheduler, t_rand, device)
                eps_rand = torch.randn_like(z0_student)

                z_t = sqrt_ab * z0_student + sqrt_1mab * eps_rand  # DDPM noising of predicted x0
                teacher_pred = teacher_unet(z_t.to(teacher_unet.dtype), t_rand, text_embeds.to(teacher_unet.dtype)).sample
                # Many SD UNets predict either eps or v; assume eps here (SD2.1 base usually eps).
                L_sds = mse(teacher_pred.to(z0_student.dtype), eps_rand)

                loss = L_perc * args.lambda_perc + L_sds * args.lambda_sds

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            pbar.set_postfix({
                "loss": float(loss.detach().cpu()),
                "L_perc": float(L_perc.detach().cpu()),
                "L_sds": float(L_sds.detach().cpu())
            })

            if global_step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"swiftedit_stage2_step{global_step}.pt")
                save_stage2_checkpoint(save_path, inverse, global_step)

            if global_step >= args.max_steps:
                break

        if global_step >= args.max_steps:
            break

    final_path = os.path.join(args.output_dir, "swiftedit_stage2_final.pt")
    save_stage2_checkpoint(final_path, inverse, global_step)
    print(f"[Stage 2] Done. Saved to {final_path}")


def parse_args():
    p = argparse.ArgumentParser(description="SwiftEdit Stage-2 (Real fine-tuning)")
    # We reuse the same base model to obtain VAE, tokenizer, text encoder, scheduler
    p.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-2-1-base")
    p.add_argument("--turbo_model", type=str, default="stabilityai/sd-turbo")
    p.add_argument("--ip_adapter_repo", type=str, default="h94/IP-Adapter")

    # SBv2 UNet (student generator) + IP weights (frozen here)
    p.add_argument("--sbv2_unet_path", type=str, required=True,
                   help="Local path to SBv2 UNet (e.g., swiftedit_weights/sbv2_0.5).")
    p.add_argument("--ip_ckpt_in", type=str, required=True,
                   help="Path to IP-Adapter weights (trained in Stage-1).")

    # Inversion init + Stage-1 ckpt (optional to overwrite)
    p.add_argument("--inverse_init", type=str, required=True,
                   help="Path used by InverseModel to init (folder containing unet_ema).")
    p.add_argument("--stage1_ckpt", type=str, default="",
                   help="(Optional) Stage-1 checkpoint .pt to load F_theta weights from.")

    # Data
    p.add_argument("--images_dir", type=str, required=True)
    p.add_argument("--prompts_file", type=str, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)

    # Train
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=8000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--mid_t", type=int, default=500, help="t for inverse UNet")
    p.add_argument("--lambda_perc", type=float, default=1.0)
    p.add_argument("--lambda_sds", type=float, default=1.0)
    p.add_argument("--sds_tmin", type=int, default=200)
    p.add_argument("--sds_tmax", type=int, default=800)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_stage2(args)
