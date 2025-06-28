import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class ImprovedGaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, T):
        super().__init__()
        self.model = model
        self.T = T
        betas = cosine_beta_schedule(T)
        self.register_buffer('betas', betas)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer('log_one_minus_alphas_bar', torch.log(1. - alphas_bar))
        self.register_buffer('log_betas', torch.log(betas))

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        model_out = self.model(x_t, t)
        pred_noise, pred_logvar = model_out.split(3, dim=1)
        # 1. MSE loss for noise prediction
        mse_loss = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=(1,2,3))
        # 2. VLB loss for variance prediction
        true_logvar = extract(self.log_betas, t, x_0.shape)
        kl = 0.5 * (true_logvar - pred_logvar + torch.exp(pred_logvar - true_logvar) - 1)
        kl = kl.mean(dim=(1,2,3))
        loss = mse_loss + kl
        return loss.mean()

class ImprovedGaussianDiffusionSampler(nn.Module):
    def __init__(self, model, T):
        super().__init__()
        self.model = model
        betas = cosine_beta_schedule(T)
        self.register_buffer('betas', betas)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.T = T

    def forward(self, x_T, multi_steps=False):
        x_t = x_T
        res_list = []
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            model_out = self.model(x_t, t)
            pred_noise, pred_logvar = model_out.split(3, dim=1)
            alphas_bar_t = extract(self.alphas_bar, t, x_t.shape)
            sqrt_recip_alphas_bar = 1. / torch.sqrt(alphas_bar_t)
            sqrt_recipm1_alphas_bar = torch.sqrt(1. / alphas_bar_t - 1)
            pred_x0 = sqrt_recip_alphas_bar * x_t - sqrt_recipm1_alphas_bar * pred_noise
            beta_t = extract(self.betas, t, x_t.shape)
            mean = (
                extract(self.alphas, t, x_t.shape).sqrt() * pred_x0 +
                (1 - extract(self.alphas, t, x_t.shape)).sqrt() * x_t
            )
            logvar = pred_logvar
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + (0.5 * logvar).exp() * noise
            if (time_step + 1) % 100 == 0 or time_step == 0:
                res_list.append(torch.clip(x_t, -1, 1))
        x_0 = x_t
        if multi_steps:
            return res_list
        else:
            return torch.clip(x_0, -1, 1)
