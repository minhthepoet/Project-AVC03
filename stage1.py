# =============================================================
#  SwiftEdit - Stage 1 (Synthetic Pretraining)
#  Copyright (c) 2025
#  Author: Nhu Duc Minh Nguyen (nnguy29)
# =============================================================

import os
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.utils import save_image

# -------------------------------
# 💠 Diffusers + Transformers
# -------------------------------
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

# -------------------------------
# 🧩 Local modules (mày đã có)
# -------------------------------
from model import InverseModel, IPSBV2Model, ImageProjModel
from losses import compute_stage1_losses
from mask_ip_controller import MaskIPController
from mask_attention_processor import MaskAttentionProcessor
from attention_processor import BasicAttentionProcessor

# -------------------------------
# ⚙️ Misc
# -------------------------------
from accelerate import Accelerator
from accelerate.utils import set_seed
from pathlib import Path
