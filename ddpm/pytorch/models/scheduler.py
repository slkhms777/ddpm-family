import torch
import numpy as np

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear'):
        self.num_timesteps = num_timesteps
        
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # 用于采样的预计算值
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, original_samples, noise, timesteps):
        """前向扩散：q(x_t|x_0)"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise