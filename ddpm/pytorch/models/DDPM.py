import torch
import torch.nn as nn
from .scheduler import NoiseScheduler

class DDPM(nn.Module):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        
    def forward(self, x0, t):
        """训练时的前向传播"""
        noise = torch.randn_like(x0)
        x_t = self.scheduler.add_noise(x0, noise, t)
        
        # UNet预测噪声
        predicted_noise = self.unet(x_t, t)
        
        return predicted_noise, noise
    
    def compute_loss(self, x0, t):
        """计算训练损失"""
        predicted_noise, target_noise = self.forward(x0, t)
        return nn.MSELoss()(predicted_noise, target_noise)