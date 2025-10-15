# SwiftEdit — Stage 2  (Real-image finetuning of F_theta)
# By minhthepoet
import argparse
import os, json, time, random
os.environ["DBG_SHAPES"] = "1"

import torch
import torch.nn as nn
from torch.optim import AdamW
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torchvision import transforms
from PIL import Image

from transformers import (
    CLIPTextModel, CLIPTokenizer,
    CLIPVisionModel, CLIPImageProcessor,
)
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

# project modules
from model import InverseModel, IPSBV2Model, ImageProjModel
from losses import PerceptualLoss, stage2_regularizer, stage2_total_loss
# Config 
parser = argparse.ArgumentParser(description="SwiftEdit Stage 2 Training (real-image finetuning)")
parser.add_argument("--model_path", type=str, default="swiftbrushV2",
                    help="Path to SwiftBrushV2 folder (SBv2 UNet/text/tokenizer)")
parser.add_argument("--teacher_model", type=str, default="stabilityai/stable-diffusion-2-1-base",
                    help="HF id/path for teacher UNet + scheduler + VAE")
parser.add_argument("--stage1_ckpt", type=str, required=True,
                    help="Path to Stage-1 checkpoint (.pth) containing inverse_net(_ema) and ip_adapter")
parser.add_argument("--data_list", type=str, required=True,
                    help="Text file with 'image_path<TAB>prompt' per line")
parser.add_argument("--save_dir", type=str, default="checkpoints_stage2",
                    help="Folder to save Stage-2 checkpoints")

parser.add_argument("--total_steps", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--lambda_perc", type=float, default=1.0)
parser.add_argument("--lambda_regu", type=float, default=1.0)
parser.add_argument("--ema", type=float, default=0.999)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
parser.add_argument("--log_every", type=int, default = 10)
parser.add_argument("--save_every", type=int, default=200)

parser.add_argument("--finetune_ip", action="store_true", help="Also fine-tune ImageProjModel in stage-2")

args = parser.parse_args()
SBV2_DIR   = args.model_path
SAVE_DIR   = args.save_dir
TOTAL_STEPS= args.total_steps
BATCH_SIZE = args.batch_size
LR         = args.lr
WD         = args.wd
EMA_DECAY  = args.ema
SEED       = args.seed

# Device + dtype
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.dtype == "bf16":
    DTYPE = torch.bfloat16
elif args.dtype == "fp16":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32

# Constants
VAE_SCALE = 0.18215
os.makedirs(SAVE_DIR, exist_ok=True)

# Helpers
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_dtype(model, device, dtype):
    return model.to(device=device, dtype=dtype)

def requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad = flag

@torch.no_grad()
def update_ema(src: nn.Module, tgt: nn.Module, decay: float = 0.999):
    for p_tgt, p_src in zip(tgt.parameters(), src.parameters()):
        p_tgt.data.mul_(decay).add_(p_src.data, alpha=1.0 - decay)

def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    return inputs.input_ids
# Load modules
def load_sbv2_unet(local_dir: str, device, dtype):
    with open(os.path.join(local_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    unet = UNet2DConditionModel.from_config(cfg)
    state = load_file(os.path.join(local_dir, "diffusion_pytorch_model.safetensors"))
    unet.load_state_dict(state, strict=False)
    requires_grad(unet, False)
    unet.eval().to(device, dtype)
    return unet

def load_teacher_components(model_id: str, device):
    teacher_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    teacher_unet.eval(); requires_grad(teacher_unet, False)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device, dtype=torch.float32)
    vae.eval(); requires_grad(vae, False)
    return teacher_unet, scheduler, vae

def load_text_encoder(device, dtype):
    local_tok = os.path.join(SBV2_DIR, "tokenizer")
    local_txt = os.path.join(SBV2_DIR, "text_encoder")
    if os.path.exists(local_tok) and os.path.exists(local_txt):
        tok = CLIPTokenizer.from_pretrained(local_tok)
        txt = CLIPTextModel.from_pretrained(local_txt)
    else:
        tok = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        txt = CLIPTextModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    txt.eval(); tok.model_max_length = 77
    txt = txt.to(device=device, dtype=dtype)
    return tok, txt

def load_image_encoder(device):
    model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    img_enc = CLIPVisionModelWithProjection.from_pretrained(model_id)
    img_proc = CLIPImageProcessor.from_pretrained(model_id)
    img_enc.eval(); img_enc.requires_grad_(False)
    return img_enc.to(device, torch.float32), img_proc
# Data utilities (real images)
class RealImagePromptDataset(torch.utils.data.Dataset):
    def __init__(self, list_file: str):
        self.items = []
        with open(list_file, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                if "\t" in ln:
                    pth, prm = ln.split("\t", 1)
                else:
                    # fallback "path||prompt"
                    pth, prm = ln.split("||", 1)
                self.items.append((pth, prm))
        self.to_vae = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(512),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        pth, prm = self.items[idx]
        img = Image.open(pth).convert("RGB")
        px = self.to_vae(img)  # [3,512,512], 0..1
        return img, px, prm, pth

def make_loader(list_file, batch_size):
    ds = RealImagePromptDataset(list_file)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda x: x[0],
    )

# Train step (Stage 2)
def train_step_stage2(
    inverse_net,         
    g_ip,                
    ip_adapter,           
    teacher_unet, scheduler, vae_teacher,
    tokenizer, text_encoder,
    img_encoder, img_processor,
    pil_imgs,  
    px_01,     
    prompts,   
    optimizer,
    perceptual,                 
    lambda_perc=1.0, lambda_regu=1.0,
    device="cuda", dtype=torch.float16
):

    from PIL import Image

    # --- determine batch size correctly ---
    B = 1 if px_01.ndim == 3 else px_01.size(0)

    # --- text cond: ensure list[str] and batch B ---
    if isinstance(prompts, str):
        prompts = [prompts]
    input_ids = tokenize_captions(tokenizer, prompts).to(device)
    text_emb  = text_encoder(input_ids)[0].to(dtype)
    if text_emb.shape[0] != B:
        text_emb = text_emb[:1].repeat(B, 1, 1)   # keep safe if B==1

    # --- image embeds (CLIP pooled): wrap PIL to list ---
    if isinstance(pil_imgs, Image.Image):
        pil_imgs = [pil_imgs]
    clip_pixels = img_processor(images=pil_imgs, return_tensors="pt")["pixel_values"].to(device, dtype=torch.float32)
    img_feats   = img_encoder(clip_pixels).image_embeds.to(dtype)  # match ip_adapter dtype

    # --- encode real images to latent z_real ---
    px_m11 = px_01.to(device, dtype=torch.float32) * 2.0 - 1.0      # [-1,1]
    if px_m11.ndim == 3:                                            # add batch dim if missing
        px_m11 = px_m11.unsqueeze(0)
    with torch.no_grad():
        z_real = vae_teacher.encode(px_m11).latent_dist.sample() * VAE_SCALE
    z_real = z_real.to(dtype)                                       # [B,4,64,64]

    # --- forward F_theta to get eps_hat ---
    eps_hat = inverse_net(z_real, text_emb)                         # [B,4,64,64]

    # --- reconstruction via frozen SBv2 (+IP tokens) ---
    t_full = torch.full((z_real.size(0),), 999, device=device, dtype=torch.long)  # use true B
    img_proj = ip_adapter(img_feats)                                 # [B,4,1024]
    cond = torch.cat([text_emb, img_proj.to(dtype)], dim=1)          # [B,81,1024]
    z_hat = g_ip.unet(eps_hat, t_full, encoder_hidden_states=cond).sample
    x_hat = vae_teacher.decode((z_hat / VAE_SCALE).to(dtype=torch.float32)).sample
    x_hat = x_hat.clamp(-1, 1)
    x_hat = (x_hat + 1.0) / 2.0
    # --- Losses ---
    # Perceptual reconstruction loss
    L_perc = perceptual(x_hat, px_01.to(device, dtype=torch.float32))

    # Regularization loss (SDS-style)
    L_regu, eps_t = stage2_regularizer(
        teacher_unet, scheduler, z_real, eps_hat,
        text_embeds=text_emb,
        tmin=200, tmax=800,
        device=device, dtype=None, use_w=True,
    )

    # Total weighted loss
    L_total, L_perc, L_reg = stage2_total_loss(
        perceptual,
        x_hat,
        px_01.to(device, dtype=torch.float32),
        L_regu,
        lambda_perc=lambda_perc,
        lambda_regu=lambda_regu,
    )
    # --- Backpropagation ---
    optimizer.zero_grad(set_to_none=True)
    L_total.backward()

    # Clip gradients to avoid explosion
    torch.nn.utils.clip_grad_norm_(inverse_net.parameters(), 1.0)
    if any(p.requires_grad for p in ip_adapter.parameters()):
        torch.nn.utils.clip_grad_norm_(ip_adapter.parameters(), 1.0)

    optimizer.step()
    # --- Return losses as floats (safe for logging) ---
    return L_total.item(), L_perc.item(), L_reg.item()


def main():
    # =============================================================
    # 0️⃣ Setup + model init
    # =============================================================
    inverse_net = to_dtype(InverseModel(pretrained_model_name_path=SBV2_DIR), DEVICE, DTYPE)
    ip_adapter  = to_dtype(ImageProjModel(), DEVICE, DTYPE)
    ema_net     = to_dtype(InverseModel(pretrained_model_name_path=SBV2_DIR), DEVICE, DTYPE)

    # load frozen UNet + teacher + encoders
    print("Loading FROZEN backbones ...")
    unet_base = load_sbv2_unet(SBV2_DIR, DEVICE, DTYPE)
    teacher_unet, scheduler, vae_t = load_teacher_components(args.teacher_model, DEVICE)
    tokenizer, text_encoder = load_text_encoder(DEVICE, DTYPE)
    img_encoder, img_processor = load_image_encoder(DEVICE)

    # Build main model
    g_ip = to_dtype(IPSBV2Model(unet_model=unet_base, image_proj_model=ip_adapter, device=DEVICE), DEVICE, DTYPE)

    # =============================================================
    # 1️⃣ Build optimizer first
    # =============================================================
    params = list(inverse_net.parameters())
    if args.finetune_ip:
        params += list(ip_adapter.parameters())
    optimizer = AdamW(params, lr=LR, weight_decay=WD)

    # EMA copy
    ema_net.load_state_dict(inverse_net.state_dict(), strict=True)
    requires_grad(ema_net, False)

    start_step = 0
    if args.resume_from is not None and os.path.exists(args.resume_from):
        print(f"Resuming Stage-2 from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=DEVICE)
        inverse_net.load_state_dict(ckpt["inverse_net"])
        ema_net.load_state_dict(ckpt["inverse_net_ema"])
        ip_adapter.load_state_dict(ckpt["ip_adapter"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed training from step {start_step}")
    else:
        print(f"Starting Stage-2 from Stage-1 checkpoint: {args.stage1_ckpt}")
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
        inverse_net.load_state_dict(ckpt["inverse_net"])
        ema_net.load_state_dict(ckpt["inverse_net"])
        ip_adapter.load_state_dict(ckpt["ip_adapter"])
        start_step = 0

    # Data + perceptual loss
    loader = make_loader(args.data_list, BATCH_SIZE)
    perceptual = PerceptualLoss(DEVICE)
    perceptual.eval()

    # Freeze non-trainables
    requires_grad(g_ip.unet, False)
    requires_grad(vae_t, False)           
    requires_grad(ip_adapter, args.finetune_ip)  

    print("--- Starting Stage-2 training ---")
    # EMA smoothing vars
    ema_L = ema_perc = ema_reg = 0.0
    alpha = 0.9
    t0 = time.time()
    step = start_step

    # Training loop
    for epoch in range(10**4):
        for batch in loader:
            if step >= TOTAL_STEPS:
                break

            pil_imgs, px_01, prompts, _ = batch
            step += 1

            L_total, L_perc, L_reg = train_step_stage2(
                inverse_net, g_ip, ip_adapter,
                teacher_unet, scheduler, vae_t,
                tokenizer, text_encoder,
                img_encoder, img_processor,
                pil_imgs, px_01, prompts,
                optimizer,
                perceptual,
                lambda_perc=args.lambda_perc, 
                lambda_regu=args.lambda_regu,
                device=DEVICE, dtype=DTYPE,
            )

            # EMA for logging
            if step == start_step:
                ema_L, ema_perc, ema_reg = L_total, L_perc, L_reg
            else:
                ema_L   = alpha * ema_L   + (1 - alpha) * L_total
                ema_perc = alpha * ema_perc + (1 - alpha) * L_perc
                ema_reg  = alpha * ema_reg  + (1 - alpha) * L_reg

            update_ema(inverse_net, ema_net, EMA_DECAY)

            if step % args.log_every == 0 or step == 1:
                dt = time.time() - t0
                print(f"[{step:06d}/{TOTAL_STEPS}] "
                      f"L={L_total:.4f} (ema={ema_L:.4f}, perc={L_perc:.4f}, ema_p={ema_perc:.4f}, "
                      f"reg={L_reg:.4f}, ema_r={ema_reg:.4f}) | {(dt/args.log_every):.3f}s/it")
                t0 = time.time()

            # Save checkpoint
            if step % args.save_every == 0 or step == TOTAL_STEPS:
                ckpt_out = {
                    "inverse_net": inverse_net.state_dict(),
                    "inverse_net_ema": ema_net.state_dict(),
                    "ip_adapter": ip_adapter.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                os.makedirs(SAVE_DIR, exist_ok=True)
                ckpt_path = os.path.join(SAVE_DIR, f"F_theta_stage2_step{step:06d}.pth")
                torch.save(ckpt_out, ckpt_path)
                print(f"💾 Saved checkpoint to {ckpt_path}")

        if step >= TOTAL_STEPS:
            break

    print("Done Stage-2 training.")
if __name__ == "__main__":
    main()
