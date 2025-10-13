# =============================================================
# SwiftEdit — Stage 1 By minhthepoet
# =============================================================
import argparse
import os, json, time, random
import torch
import torch.nn as nn
from torch.optim import AdamW
from safetensors.torch import load_file
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import (
    CLIPTextModel, CLIPTokenizer,
    CLIPVisionModel, CLIPImageProcessor,
)
from model import InverseModel, IPSBV2Model, ImageProjModel
from losses import stage1_loss

# =============================================================
# Config 
# =============================================================

parser = argparse.ArgumentParser(description="SwiftEdit Stage 1 Training")
parser.add_argument("--model_path", type=str, default="swiftbrushV2", help="Path to SwiftBrushV2 folder")
parser.add_argument("--prompt_file", type=str, default="journeydb_cache/journey_prompts.txt", help="Path to prompt list")
parser.add_argument("--save_dir", type=str, default="checkpoints_stage1", help="Folder to save checkpoints")
parser.add_argument("--total_steps", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--lambda_regr", type=float, default=1.0)
parser.add_argument("--ema", type=float, default=0.999)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
args = parser.parse_args()

# Assign configs
PROMPT_FILE = args.prompt_file
SBV2_DIR = args.model_path
SAVE_DIR = args.save_dir
TOTAL_STEPS = args.total_steps
BATCH_SIZE = args.batch_size
LR = args.lr
WD = args.wd
LAMBDA = args.lambda_regr
EMA_DECAY = args.ema
SEED = args.seed

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



# =============================================================
# Helpers
# =============================================================

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


# =============================================================
# Load modules
# =============================================================

def load_sbv2_unet(local_dir: str, device, dtype):
    with open(os.path.join(local_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    unet = UNet2DConditionModel.from_config(cfg)
    state = load_file(os.path.join(local_dir, "diffusion_pytorch_model.safetensors"))
    unet.load_state_dict(state, strict=False)
    requires_grad(unet, False)
    unet.eval().to(device, dtype)
    return unet

def load_vae(device, dtype):
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")
    requires_grad(vae, False)
    vae.eval().to(device, dtype)
    return vae

def load_text_encoder(device, dtype):
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    txt = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    requires_grad(txt, False)
    txt.eval().to(device, dtype)
    return tok, txt

def load_image_encoder(device, dtype):
    img_enc = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    img_proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    requires_grad(img_enc, False)
    img_enc.eval().to(device, dtype)
    return img_enc, img_proc


# =============================================================
# Data utilities
# =============================================================

def clip_preprocess_from_tensor(x_rgb_float, img_proc: CLIPImageProcessor):
    x = (x_rgb_float.clamp(-1, 1) + 1.0) / 2.0
    x = x.detach().cpu()
    pil_list = []
    for i in range(x.size(0)):
        pil = img_proc.postprocess(x[i].unsqueeze(0), output_type="pil")[0]
        pil_list.append(pil)
    pixel = img_proc(images=pil_list, return_tensors="pt")["pixel_values"]
    return pixel


@torch.no_grad()
def generate_synthetic_batch(
    batch_size,
    prompts,
    tokenizer,
    text_encoder,
    unet_base,
    vae,
    img_encoder,
    img_processor,
    device,
    dtype,
):
    texts = [prompts[random.randrange(len(prompts))] for _ in range(batch_size)]
    tokens = tokenizer(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    text_emb = text_encoder(**tokens).last_hidden_state

    eps = torch.randn(batch_size, 4, 64, 64, device=device, dtype=dtype)
    t = torch.full((batch_size,), 999, device=device, dtype=torch.long)
    latent_pred = unet_base(eps, t, encoder_hidden_states=text_emb).sample
    x_synth = vae.decode(latent_pred / VAE_SCALE).sample
    z = vae.encode(x_synth).latent_dist.sample() * VAE_SCALE

    pixel_values = clip_preprocess_from_tensor(x_synth, img_processor).to(device)
    img_feats = img_encoder(pixel_values=pixel_values).last_hidden_state

    return z, eps, text_emb, img_feats, texts


# =============================================================
# Train step
# =============================================================

def train_step(inverse_net, ip_model, g_ip, optimizer, batch, lambda_regr, device, dtype):
    z, eps, text_emb, img_feats, _ = batch
    eps_hat = inverse_net(z, text_emb)
    img_proj = ip_model(img_feats)
    t = torch.full((z.size(0),), 999, device=device, dtype=torch.long)
    z_hat = g_ip.unet(eps_hat, t, encoder_hidden_states=torch.cat([text_emb, img_proj], dim=1)).sample
    L_rec, L_regr, L_total = stage1_loss(z, z_hat, eps, eps_hat, lambda_regr)
    optimizer.zero_grad(set_to_none=True)
    L_total.backward()
    optimizer.step()
    return L_rec.detach(), L_regr.detach(), L_total.detach()


# =============================================================
# Main training loop
# =============================================================

def main():
    set_seed(SEED)

    with open(PROMPT_FILE, "r") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(prompts)} prompts.")

    print("Loading FROZEN backbones ...")
    unet_base = load_sbv2_unet(SBV2_DIR, DEVICE, DTYPE)
    vae = load_vae(DEVICE, DTYPE)
    tokenizer, text_encoder = load_text_encoder(DEVICE, DTYPE)
    img_encoder, img_processor = load_image_encoder(DEVICE, DTYPE)

    print("🔹 Initializing TRAINABLE models (f_the + IP-Adapter) ...")
    inverse_net = to_dtype(InverseModel(), DEVICE, DTYPE)
    ip_adapter  = to_dtype(ImageProjModel(), DEVICE, DTYPE)
    g_ip        = to_dtype(IPSBV2Model(unet_model=unet_base, image_proj_model=ip_adapter), DEVICE, DTYPE)

    ema_net = to_dtype(InverseModel(), DEVICE, DTYPE)
    ema_net.load_state_dict(inverse_net.state_dict(), strict=True)
    requires_grad(ema_net, False)

    optimizer = AdamW(list(inverse_net.parameters()) + list(ip_adapter.parameters()), lr=LR, weight_decay=WD)

    print("---Starting Stage-1 training ...")
    t0 = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        batch = generate_synthetic_batch(
            BATCH_SIZE, prompts, tokenizer, text_encoder, unet_base,
            vae, img_encoder, img_processor, DEVICE, DTYPE
        )
        L_rec, L_regr, L_total = train_step(inverse_net, ip_adapter, g_ip, optimizer, batch, LAMBDA, DEVICE, DTYPE)
        update_ema(inverse_net, ema_net, EMA_DECAY)

        if step % 20 == 0 or step == 1:
            dt = time.time() - t0
            print(f"[{step:04d}/{TOTAL_STEPS}] L={L_total.item():.4f} "
                  f"(rec={L_rec.item():.4f}, regr={L_regr.item():.4f}) | {dt/20:.2f}s/20steps")
            t0 = time.time()

        if step % 1000 == 0 or step == TOTAL_STEPS:
            ckpt = {
                "inverse_net": inverse_net.state_dict(),
                "inverse_net_ema": ema_net.state_dict(),
                "ip_adapter": ip_adapter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            out_path = os.path.join(SAVE_DIR, f"F_theta_stage1_step{step:06d}.pth")
            torch.save(ckpt, out_path)
            print(f"Saved checkpoint to {out_path}")

    print("Done Stage-1 training.")


if __name__ == "__main__":
    main()
