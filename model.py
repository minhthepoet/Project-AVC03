import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPVisionModelWithProjection,
)

from src.mask_ip_controller import *
from src.attention_processor import AttnProcessor2_0 as AttnProcessor
from src.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor
from src.mask_attention_processor import IPAttnProcessor2_0WithIPMaskController

def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(
        self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4
    ):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class InverseModel(nn.Module):
    """
        Inversion Network that bring source image latents to noisy latents.
    """
    def __init__(
        self, 
        pretrained_model_name_path, 
        model_name="stabilityai/sd-turbo",
        dtype="fp32",
        device="cuda"
    ):
        super().__init__()
        if dtype == "fp16":
            self.weight_dtype = torch.float16
        elif dtype == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32

        self.device = device
        self.model_name = model_name
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae").to(
            self.device, dtype=torch.float32
        )

        unet_path = os.path.join(pretrained_model_name_path, "unet_ema")
        if not os.path.exists(os.path.join(unet_path, "config.json")):
            unet_path = pretrained_model_name_path  
        self.unet_inverse = UNet2DConditionModel.from_pretrained(
            unet_path
        ).to(self.device, dtype=self.weight_dtype)

        self.unet_inverse.eval()
        local_tok_path = os.path.join(pretrained_model_name_path, "tokenizer")
        local_txt_path = os.path.join(pretrained_model_name_path, "text_encoder")

        if os.path.exists(local_tok_path) and os.path.exists(local_txt_path):
            self.tokenizer = CLIPTokenizer.from_pretrained(local_tok_path)
            self.text_encoder = CLIPTextModel.from_pretrained(local_txt_path).to(
                self.device, dtype=self.weight_dtype
            )
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self.text_encoder = CLIPTextModel.from_pretrained(
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            ).to(self.device, dtype=self.weight_dtype)

        T = torch.ones((1,), dtype=torch.int64, device=self.device)
        T = T * (self.noise_scheduler.config.num_train_timesteps - 1)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        self.corrupt_alpha_t = (alphas_cumprod[int(T / 4)] ** 0.5).view(-1, 1, 1, 1)
        self.corrupt_sigma_t = ((1 - alphas_cumprod[int(T / 4)]) ** 0.5).view(-1, 1, 1, 1)

        del alphas_cumprod

    def forward(self, z, text_emb):
        """
        z         : latent từ VAE (shape [B,4,64,64])
        text_emb  : text embedding từ CLIP (shape [B,77,1024])
        return    : eps_hat (predicted noise)
        """
        t = torch.full(
            (z.size(0),), 
            self.noise_scheduler.config.num_train_timesteps - 1,
            device=self.device, 
            dtype=torch.long
        )

        eps_hat = self.unet_inverse(
            z, 
            t, 
            encoder_hidden_states=text_emb
        ).sample

        return eps_hat


class AuxiliaryModel:
    """
        A few auxiliary and supported models (text encoder, noise scheduler, tokenizer, ...) as separate modules.
    """
    def __init__(
        self,
        model_name="stabilityai/stable-diffusion-2-1-base",
        image_encoder_path="h94/IP-Adapter",
        device="cuda",
    ):
        self.device = device
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(
            self.device, dtype=torch.float32
        )

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path, subfolder="models/image_encoder"
        ).to(device, dtype=torch.float32)
        self.image_encoder.requires_grad_(False)

        self.clip_image_processor = CLIPImageProcessor()


class IPSBV2Model(torch.nn.Module):
    def __init__(
        self,
        unet_model,          
        image_proj_model,        
        device="cuda",
        dtype=torch.float32
    ):
        super().__init__()
        self.device = device
        self.unet = unet_model.to(device).eval()      
        self.image_proj_model = image_proj_model.to(device)
        self.dtype = dtype
    def forward(self, eps_hat, t, text_emb, img_feats):
        img_proj = self.image_proj_model(img_feats)           # [B, 4, 1024]
        cond = torch.cat([text_emb, img_proj], dim=1)         # concat condition tokens
        out = self.unet(eps_hat, t, encoder_hidden_states=cond).sample
        return out
