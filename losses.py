# losses.py
import torch
import torch.nn as nn

# ---------- Perceptual ----------
class PerceptualLoss(nn.Module):
    """Prefer DISTS, fallback LPIPS, else L1."""
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
        # x,y in [0,1], shape [B,3,H,W]
        if self.mode == "DISTS":
            return self.impl(x, y)
        elif self.mode == "LPIPS":
            return self.impl(x*2-1, y*2-1).mean()
        else:
            return self.impl(x, y)

# ---------- Stage-1 ----------
def stage1_loss(eps_hat, eps, z0_student, z0_target, lambda_regr=1.0):
    mse = nn.MSELoss()
    L_regr = mse(eps_hat, eps)
    L_rec  = mse(z0_student, z0_target)
    return L_rec + lambda_regr * L_regr, L_rec.detach(), L_regr.detach()

# ---------- SDS utilities ----------
def _alpha_bar_terms(scheduler, t_tensor: torch.Tensor, device):
    alphas_cumprod = scheduler.alphas_cumprod.to(device)  # [T]
    a_bar = alphas_cumprod[t_tensor]                      # [B]
    sqrt_ab = a_bar.sqrt().view(-1,1,1,1)
    sqrt_1m = (1.0 - a_bar).sqrt().view(-1,1,1,1)
    return sqrt_ab, sqrt_1m

@torch.no_grad()
def _sample_teacher_eps(teacher_unet, z_t, t, text_embeds):
    # SD2.1 UNet predicts eps by default
    return teacher_unet(z_t, t, text_embeds).sample

# ---------- Stage-2 SDS regularizer ----------
def sds_regularizer(teacher_unet, scheduler, z0_pred, text_embeds,
                    tmin=200, tmax=800, device="cuda", dtype=None):
    B = z0_pred.size(0)
    t_rand = torch.randint(low=tmin, high=tmax+1, size=(B,), device=device, dtype=torch.int64)
    sqrt_ab, sqrt_1m = _alpha_bar_terms(scheduler, t_rand, device)
    eps_rand = torch.randn_like(z0_pred)
    z_t = sqrt_ab * z0_pred + sqrt_1m * eps_rand
    te_dtype = next(teacher_unet.parameters()).dtype if dtype is None else dtype
    teacher_pred = _sample_teacher_eps(teacher_unet, z_t.to(te_dtype), t_rand, text_embeds.to(te_dtype))
    mse = nn.MSELoss()
    L_sds = mse(teacher_pred.to(z0_pred.dtype), eps_rand)
    return L_sds, t_rand

# ---------- Stage-2 total ----------
def stage2_total_loss(perceptual_loss, x_hat, x_gt, L_sds, lambda_perc=1.0, lambda_sds=1.0):
    L_perc = perceptual_loss(x_hat, x_gt)
    return lambda_perc * L_perc + lambda_sds * L_sds, L_perc.detach()
