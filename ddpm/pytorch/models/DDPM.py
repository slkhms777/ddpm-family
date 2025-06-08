import torch
import torch.nn as nn
from .scheduler import NoiseScheduler
from .unet import UNet
"""
UNet(
    in_channels=3,              # 输入通道数 (RGB图像)
    out_ch=3,                   # 输出通道数 (RGB图像，fixedlarge方差时)
    ch=128,                     # 基础通道数
    ch_mult=(1, 2, 2, 2),      # 通道倍数：[128, 256, 256, 256]
    num_res_blocks=2,           # 每层ResNet块数量
    attn_resolutions=(16,),     # 在16x16分辨率使用注意力
    dropout=0.1,                # Dropout率
    resolution=32,              # 输入图像分辨率 (CIFAR-10)
    use_timestep=True           # 使用时间步条件化
)

num_diffusion_timesteps=1000    # 扩散时间步数 T
beta_start=0.0001              # 初始噪声方差 β₁
beta_end=0.02                  # 最终噪声方差 βₜ  
beta_schedule='linear'         # 线性噪声调度
model_mean_type='eps'          # 预测噪声 ε 而非均值 μ
model_var_type='fixedlarge'    # 固定方差（不学习）
loss_type='mse'                # MSE损失函数
"""

class DDPM(nn.Module):
    def __init__(
        self,
        unet_config=None,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='linear'
    ):
        super().__init__()
        
        # 默认UNet配置
        if unet_config is None:
            unet_config = {
                'in_channels': 3,
                'out_ch': 3,
                'ch': 128,
                'ch_mult': (1, 2, 2, 2),
                'num_res_blocks': 2,
                'attn_resolutions': (16,),
                'dropout': 0.1,
                'resolution': 32,
                'use_timestep': True
            }
        
        self.unet = UNet(**unet_config)
        
        self.scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule
        )
        
    def forward(self, x, t=None):
        """前向传播 - 用于训练"""
        if t is None:
            # 训练时随机采样时间步
            batch_size = x.shape[0]
            t = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=x.device)
        
        return self.unet(x, t)
    
    def training_step(self, x_0):
        """训练步骤"""
        return self.scheduler.training_loss(self.unet, x_0)
    
    def sample(self, shape, device='cpu'):
        """生成样本"""
        return self.scheduler.sample(self.unet, shape, device)

