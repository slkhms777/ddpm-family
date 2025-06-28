
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss

class DDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_bar', torch.cumprod(self.alphas, dim=0))

    def forward(self, x_T, ddim_timesteps, eta=0.0, multi_steps=False):
        """
        DDIM sampling algorithm.
        """
        times = torch.linspace(-1, self.T - 1, steps=ddim_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_t = x_T
        res_list = []
        for time, time_next in time_pairs:
            t_tensor = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time
            
            alphas_bar_t = extract(self.alphas_bar, t_tensor, x_t.shape)
            alphas_bar_t_next = extract(self.alphas_bar, t_tensor.new_full(t_tensor.shape, time_next), x_t.shape)

            eps = self.model(x_t, t_tensor)
            
            pred_x0 = (x_t - torch.sqrt(1 - alphas_bar_t) * eps) / torch.sqrt(alphas_bar_t)
            
            if eta > 0:
                sigma_t = eta * torch.sqrt((1 - alphas_bar_t_next) / (1 - alphas_bar_t) * (1 - alphas_bar_t / alphas_bar_t_next))
            else:
                sigma_t = 0.

            pred_dir_xt = torch.sqrt(1 - alphas_bar_t_next - sigma_t**2) * eps
            
            x_t_next = torch.sqrt(alphas_bar_t_next) * pred_x0 + pred_dir_xt
            if eta > 0:
                x_t_next += sigma_t * torch.randn_like(x_t)

            x_t = x_t_next
            
            if multi_steps and (time_next < 100 or time_next % 10 == 0):
                 res_list.append(torch.clip(x_t, -1, 1))

        x_0 = x_t
        if multi_steps:
            return res_list
        else:
            return torch.clip(x_0, -1, 1)
