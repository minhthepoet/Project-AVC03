# losses.py by minhthepoet
import torch
import torch.nn as nn

def stage1_loss(z, z_hat, eps, eps_hat, lambda_regr=1.0):
    mse = nn.MSELoss()
    L_rec  = mse(z_hat, z)
    L_regr = mse(eps_hat, eps)
    L_total = L_rec + lambda_regr * L_regr
    return L_rec, L_regr, L_total

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
        if self.mode == "DISTS":
            return self.impl(x, y)
        elif self.mode == "LPIPS":
            return self.impl(x * 2 - 1, y * 2 - 1).mean()
        else:
            return self.impl(x, y)



def _alpha_bar_terms(scheduler, t_tensor: torch.Tensor, device=None, dtype=None):
    a_cum = scheduler.alphas_cumprod  
    if device is None:
        device = a_cum.device
    T = a_cum.shape[0]

    if t_tensor.dtype != torch.long:
        t_idx = (t_tensor.clamp(0, 1) * (T - 1)).long()
    else:
        t_idx = t_tensor.clamp(0, T - 1)

    a_bar = a_cum.to(device)[t_idx]           
    if dtype is not None:
        a_bar = a_bar.to(dtype)

    a_bar = a_bar.view(-1, 1, 1, 1)            
    sqrt_ab = a_bar.sqrt()
    sqrt_1m = (1.0 - a_bar).sqrt()
    return sqrt_ab, sqrt_1m, a_bar


def stage2_regularizer(teacher_unet, scheduler, z_real, eps_hat, text_embeds,
                       tmin=200, tmax=800, device="cuda", dtype=None, use_w=True):
    B = eps_hat.size(0)
    t = torch.randint(low=tmin, high=tmax + 1, size=(B,), device=device, dtype=torch.int64)
    sqrt_ab, sqrt_1m = _alpha_bar_terms(scheduler, t, device)

    z_t = sqrt_ab * z_real + sqrt_1m * eps_hat
    te_dtype = next(teacher_unet.parameters()).dtype if dtype is None else dtype
    with torch.no_grad():
        eps_teacher = teacher_unet(z_t.to(te_dtype), t, text_embeds.to(te_dtype)).sample
    eps_teacher = eps_teacher.to(eps_hat.dtype)
    if use_w:
        a_bar = scheduler.alphas_cumprod.to(device)[t].view(-1,1,1,1) 
        w = (1.0 - a_bar)
        L_regu = (w * (eps_hat - eps_teacher).pow(2)).mean()
    else:
        L_regu = (eps_hat - eps_teacher).pow(2).mean()

    return L_regu, t


def stage2_total_loss(perceptual_loss, x_hat, x_gt, L_regu, lambda_perc=1.0, lambda_regu=1.0):
    L_perc = perceptual_loss(x_hat, x_gt)
    L_total = lambda_perc * L_perc - lambda_regu * L_regu
    return L_total, L_perc, L_regu
