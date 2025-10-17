# =============================================================
# SwiftEdit — Inference Script (One-step editing)
# =============================================================
# Copyright (c) Qualcomm Technologies, Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Author: Adapted by Nhu Duc Minh Nguyen (nnguy29@uic.edu)
# =============================================================

import os, time
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from models_infer import *
import argparse

# Argparse
parser = argparse.ArgumentParser(description="SwiftEdit Inference Script")

parser.add_argument(
    "--weights_root",
    type=str,
    default="swiftedit_weights",
    help="Path to the folder containing all SwiftEdit weights (inverse_ckpt, ip_adapter_ckpt, sbv2_0.5).",
)

parser.add_argument(
    "--img_path",
    type=str,
    default="./assets/imgs_demo/woman_face.jpg",
    help="Path to the source image for editing.",
)

parser.add_argument(
    "--src_p",
    type=str,
    default="woman",
    help="Source prompt describing the original image.",
)

parser.add_argument(
    "--edit_p",
    type=str,
    default="Taylor Swift",
    help="Target prompt describing the desired edit.",
)

parser.add_argument(
    "--output_path",
    type=str,
    default=None,
    help="Path to save the edited image output (auto if None).",
)

parser.add_argument(
    "--mask_threshold",
    type=float,
    default=0.5,
    help="Threshold for binary mask generation between source and target prompts.",
)

parser.add_argument(
    "--scale_ta",
    type=float,
    default=1.0,
    help="Scaling factor for text alignment strength in IP-Adapter.",
)

parser.add_argument(
    "--scale_edit",
    type=float,
    default=0.2,
    help="Scaling factor for edited regions (foreground influence).",
)

parser.add_argument(
    "--scale_non_edit",
    type=float,
    default=1.0,
    help="Scaling factor for background preservation (non-edited regions).",
)

parser.add_argument(
    "--clamp_rate",
    type=float,
    default=3.0,
    help="Clamping factor to stabilize the automatic mask generation.",
)

args = parser.parse_args()

# Helper function
def to_binary(pix, threshold=0.5):
    return 1.0 if float(pix) > threshold else 0.0

# Main editing function
@torch.no_grad()
def edit_image(
    img_path,
    src_p,
    edit_p,
    inverse_model,
    aux_model,
    ip_sb_model,
    scale_ta=1.0,
    scale_edit=0.2,
    scale_non_edit=1.0,
    clamp_rate=3.0,
    mask_threshold=0.5,
):
    """
    Perform one-step text-guided image editing.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mid_timestep = torch.ones((1,), dtype=torch.int64, device=device) * 500

    # Load and preprocess input image
    pil_img_cond = Image.open(img_path).convert("RGB").resize((512, 512))
    processed_image = to_tensor(pil_img_cond).unsqueeze(0).to(device) * 2 - 1

    # Encode latent
    latents = inverse_model.vae.encode(
        processed_image.to(inverse_model.weight_dtype)
    ).latent_dist.sample()
    latents = latents * inverse_model.vae.config.scaling_factor
    dub_latents = torch.cat([latents] * 2, dim=0)

    # Text embeddings
    input_id = tokenize_captions(inverse_model.tokenizer, [src_p, edit_p]).to(device)
    encoder_hidden_state = inverse_model.text_encoder(input_id)[0].to(
        dtype=inverse_model.weight_dtype
    )

    # Predict inverted noise
    predict_inverted_code = inverse_model.unet_inverse(
        dub_latents, mid_timestep, encoder_hidden_state
    ).sample.to(device, dtype=inverse_model.weight_dtype)

    # Compute mask difference between source/edit
    inverted_noise_1, inverted_noise_2 = predict_inverted_code.chunk(2)
    subed = (inverted_noise_1 - inverted_noise_2).abs_().mean(dim=[0, 1])
    max_v = (subed.mean() * clamp_rate).item()
    mask12 = subed.clamp(0, max_v) / max_v
    mask12 = mask12.detach().cpu().apply_(lambda pix: to_binary(pix, mask_threshold)).to(device)

    # Combine latent and predicted noise
    input_sb = ip_sb_model.alpha_t * latents + ip_sb_model.sigma_t * inverted_noise_1

    # Mask-aware editing control
    mask_controller = MaskController(
        mask12,
        scale_text_hiddenstate=scale_ta,
        scale_ip_fg=scale_edit,
        scale_ip_bg=scale_non_edit,
    )
    ip_sb_model.set_controller(mask_controller, where=["mid_blocks", "up_blocks"])

    # Generate final edited image
    res_gen_img, _ = ip_sb_model.gen_img(
        pil_image=pil_img_cond, prompts=[src_p, edit_p], noise=input_sb
    )

    return res_gen_img

# Entry point
if __name__ == "__main__":
    print(" Loading models...")

    # Load models and weights
    inverse_ckpt = os.path.join(args.weights_root, "inverse_ckpt-120k")
    inverse_model = InverseModel(inverse_ckpt)
    aux_model = AuxiliaryModel()

    path_unet_sb = os.path.join(args.weights_root, "sbv2_0.5")
    ip_ckpt = os.path.join(args.weights_root, "ip_adapter_ckpt-90k/ip_adapter.bin")
    ip_sb_model = IPSBV2Model(
        path_unet_sb, ip_ckpt, aux_model, with_ip_mask_controller=True
    )

    print(f" Models loaded from: {args.weights_root}")

    # Run inference
    start_time = time.time()
    result = edit_image(
        args.img_path,
        args.src_p,
        args.edit_p,
        inverse_model,
        aux_model,
        ip_sb_model,
        scale_ta=args.scale_ta,
        scale_edit=args.scale_edit,
        scale_non_edit=args.scale_non_edit,
        clamp_rate=args.clamp_rate,
        mask_threshold=args.mask_threshold,
    )

    duration = time.time() - start_time
    print(f" Edit completed: '{args.src_p}' -> '{args.edit_p}' in {duration:.2f}s")

    # Save result
    if args.output_path is None:
        # auto-generate name
        safe_src = args.src_p.replace(" ", "_")
        safe_edit = args.edit_p.replace(" ", "_")
        args.output_path = f"result_{safe_src}_to_{safe_edit}.png"

    save_image(result, args.output_path)
    print(f" Saved edited image at: {args.output_path}")
