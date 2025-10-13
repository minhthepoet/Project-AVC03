# train_stage1.py by minhthepoet
# Copyright (c) 2025
# SwiftEdit - Stage 1 (Synthetic pretraining) training script
#
# This script trains the Inversion Network F_theta together with the IP-Adapter branch
# using synthetic supervision, following Section 4.1 (Stage 1) of the SwiftEdit paper.
#
# Requirements:
#   - The same repo structure you showed (models.py, src/* present)
#   - Hugging Face diffusers/transformers installed per requirements.txt
#   - A folder of conditioning images (used to encode latents and image features)
#   - A text file of prompts (one prompt per line)
#
# Output:
#   - A checkpoint .pt with keys: "unet_inverse", "image_proj_model", "adapter_modules"
#
# Notes:
#   - We freeze the SBv2 UNet (generator) and all aux modules.
#   - We train: F_theta (unet_inverse), image_proj_model, adapter_modules.
#   - Teacher path (to get z0_target) is computed under torch.no_grad().
#   - By default, we use mid_timestep = 500 and final_timestep = 999 (as in infer.py).

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import resize, to_pil_image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from model import AuxiliaryModel, IPSBV2Model, InverseModel, tokenize_captions


# -----------------------------
# Dataset
# -----------------------------
class ImagePromptDataset(Dataset):
    def __init__(self, images_dir: str, prompts_file: str, image_size: int = 512):
        self.images = []
        p = Path(images_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            self.images += list(p.glob(ext))
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {images_dir}")

        with open(prompts_file, "r", encoding="utf-8") as f:
            self.prompts = [ln.strip() for ln in f if ln.strip()]
        if len(self.prompts) == 0:
            raise RuntimeError(f"No prompts found in {prompts_file}")

        self.image_size = image_size

    def __len__(self):
        return max(len(self.images), len(self.prompts))

    def __getitem__(self, idx):
        img_path = self.images[idx % len(self.images)]
        prompt = self.prompts[idx % len(self.prompts)]
        img = read_image(str(img_path)).float() / 255.0  # [C,H,W] in [0,1]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        img = resize(img, [self.image_size, self.image_size],
                     interpolation=InterpolationMode.BICUBIC, antialias=True)
        return img, prompt


def collate_fn(batch):
    imgs, prompts = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # [B,3,H,W]
    return imgs, list(prompts)


# -----------------------------
# Utils
# -----------------------------
def requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


def save_stage1_checkpoint(save_path: str, inverse_model: InverseModel, ip_sb_model: IPSBV2Model, step: int):
    ckpt = {
        "step": step,
        "unet_inverse": inverse_model.unet_inverse.state_dict(),
        "image_proj_model": ip_sb_model.image_proj_model.state_dict(),
        "adapter_modules": {name: m.state_dict() for name, m in ip_sb_model.unet.attn_processors.items()},
    }
    torch.save(ckpt, save_path)


# -----------------------------
# Training
# -----------------------------
def train_stage1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Aux pack (scheduler, VAE, text/image encoders)
    aux = AuxiliaryModel(
        model_name=args.base_model,            # e.g. "stabilityai/stable-diffusion-2-1-base"
        image_encoder_path=args.ip_adapter_repo,  # e.g. "h94/IP-Adapter"
        device=str(device),
    )

    # 2) SBv2 + IP-Adapter (generator side)
    ip_sb = IPSBV2Model(
        pretrained_model_name_path=args.sbv2_unet_path,          # local folder, e.g. swiftedit_weights/sbv2_0.5
        ip_model_path=args.ip_ckpt_in,                           # e.g. swiftedit_weights/ip_adapter_ckpt-90k/ip_adapter.bin
        aux_model=aux,
        device=str(device),
        with_ip_mask_controller=False,
    ).to(device)

    # 3) Inversion Network Fθ
    inverse = InverseModel(
        pretrained_model_name_path=args.inverse_init,            # folder with subfolder "unet_ema", e.g. swiftedit_weights/inverse_ckpt-120k
        model_name=args.turbo_model,                             # "stabilityai/sd-turbo"
        dtype="fp16" if args.fp16 else "fp32",
        device=str(device),
    )

    # 4) Freeze / Unfreeze
    ip_sb.unet.requires_grad_(False)                 # freeze SBv2 UNet
    requires_grad(ip_sb.image_proj_model, True)      # train projection head
    for m in ip_sb.unet.attn_processors.values():    # train IP-Adapter processors
        requires_grad(m, True)

    requires_grad(aux.vae, False)
    requires_grad(aux.text_encoder, False)
    requires_grad(aux.image_encoder, False)

    inverse.unet_inverse.train()
    requires_grad(inverse.unet_inverse, True)

    # 5) Optimizer
    params = list(inverse.unet_inverse.parameters()) \
           + list(ip_sb.image_proj_model.parameters())
    for m in ip_sb.unet.attn_processors.values():
        params += list(m.parameters())

    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # 6) Data
    ds = ImagePromptDataset(args.images_dir, args.prompts_file, image_size=args.resolution)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=True, collate_fn=collate_fn, drop_last=True)

    # 7) Timesteps
    mid_timestep = torch.ones((1,), dtype=torch.int64, device=device) * args.mid_t  # 500

    # 8) Loop
    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    mse = nn.MSELoss()

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Stage1 Epoch {epoch+1}/{args.epochs}", leave=False)
        for imgs, prompts in pbar:
            imgs = imgs.to(device)

            with torch.no_grad():
                # Encode source images -> latents
                latents = aux.vae.encode(imgs.to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * aux.vae.config.scaling_factor  # [B,4,H/8,W/8]

            # Ground-truth noise
            eps = torch.randn_like(latents, device=device)

            # Text & image embeds
            with torch.no_grad():
                input_ids = tokenize_captions(aux.tokenizer, prompts).to(device)
                text_embeds = aux.text_encoder(input_ids)[0].to(dtype=torch.float32)

                pil_images = [to_pil_image(img.detach().cpu()) for img in imgs]
                clip_image = aux.clip_image_processor(images=pil_images, return_tensors="pt").pixel_values
                clip_image_embeds = aux.image_encoder(clip_image.to(device, dtype=torch.float32)).image_embeds
                image_prompt_embeds = ip_sb.image_proj_model(clip_image_embeds)  # [B, extra, 1024]
                prompt_embeds = torch.cat([text_embeds, image_prompt_embeds], dim=1)

            alpha_t = ip_sb.alpha_t
            sigma_t = ip_sb.sigma_t

            # Teacher target z0 (no grad)
            with torch.no_grad():
                noise_teacher = alpha_t * latents + sigma_t * eps
                model_pred_teacher = ip_sb.unet(noise_teacher, ip_sb.timestep, prompt_embeds).sample
                if model_pred_teacher.shape[1] == noise_teacher.shape[1] * 2:
                    model_pred_teacher, _ = torch.split(model_pred_teacher, noise_teacher.shape[1], dim=1)
                z0_target = (noise_teacher - sigma_t * model_pred_teacher) / alpha_t  # [B,4,H/8,W/8]

            # Student path
            with torch.cuda.amp.autocast(enabled=args.fp16):
                eps_hat = inverse.unet_inverse(latents, mid_timestep, text_embeds).sample  # [B,4,H/8,W/8]
                noise_student = alpha_t * latents + sigma_t * eps_hat
                model_pred_student = ip_sb.unet(noise_student, ip_sb.timestep, prompt_embeds).sample
                if model_pred_student.shape[1] == noise_student.shape[1] * 2:
                    model_pred_student, _ = torch.split(model_pred_student, noise_student.shape[1], dim=1)
                z0_student = (noise_student - sigma_t * model_pred_student) / alpha_t

                L_regr = mse(eps_hat, eps)
                L_rec  = mse(z0_student, z0_target)
                loss   = L_rec + args.lambda_regr * L_regr

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            pbar.set_postfix({
                "loss": float(loss.detach().cpu()),
                "L_rec": float(L_rec.detach().cpu()),
                "L_regr": float(L_regr.detach().cpu())
            })

            if global_step % args.save_every == 0:
                save_path = os.path.join(args.output_dir, f"swiftedit_stage1_step{global_step}.pt")
                save_stage1_checkpoint(save_path, inverse, ip_sb, global_step)

            if global_step >= args.max_steps:
                break

        if global_step >= args.max_steps:
            break

    final_path = os.path.join(args.output_dir, f"swiftedit_stage1_final.pt")
    save_stage1_checkpoint(final_path, inverse, ip_sb, global_step)
    print(f"[Stage 1] Done. Saved to {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="SwiftEdit Stage 1 (Synthetic) Training")
    parser.add_argument("--sbv2_unet_path", type=str, required=True,
                        help="Local path to SBv2 UNet folder (e.g., swiftedit_weights/sbv2_0.5).")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-2-1-base",
                        help="HF id or local path for SD2.1 base (provides VAE, tokenizer, text encoder, scheduler).")
    parser.add_argument("--turbo_model", type=str, default="stabilityai/sd-turbo",
                        help="HF id or local path used by InverseModel for tokenizer/text encoder init.")
    parser.add_argument("--ip_adapter_repo", type=str, default="h94/IP-Adapter",
                        help="HF repo for IP-Adapter image encoder.")
    parser.add_argument("--ip_ckpt_in", type=str, required=True,
                        help="Path to initialize IP-Adapter weights (e.g., swiftedit_weights/ip_adapter_ckpt-90k/ip_adapter.bin)")
    parser.add_argument("--inverse_init", type=str, required=True,
                        help="Path to folder that contains subfolder `unet_ema` to initialize inverse UNet (e.g., swiftedit_weights/inverse_ckpt-120k)")
    parser.add_argument("--images_dir", type=str, default="assets/imgs_demo", help="Folder of conditioning images.")
    parser.add_argument("--prompts_file", type=str, required=True, help="Text file of prompts (one per line).")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_regr", type=float, default=1.0, help="Weight for epsilon regression loss.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="swiftedit_stage1_out")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision.")
    parser.add_argument("--mid_t", type=int, default=500, help="Mid timestep for inversion UNet (match infer.py).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_stage1(args)
