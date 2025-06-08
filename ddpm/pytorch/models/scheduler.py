import torch
import torch.nn as nn
import numpy as np

class NoiseScheduler(nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear'):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # 注册为buffer，PyTorch会自动处理设备移动
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
    def add_noise(self, original_samples, noise, timesteps):
        """前向扩散：q(x_t|x_0)"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].flatten()
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].flatten()
        
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def step(self, model_output, timestep, sample):
        """
        单步反向扩散采样：p(x_{t-1}|x_t)
        
        Args:
            model_output: 模型预测的噪声 ε_θ(x_t, t)
            timestep: 当前时间步 t
            sample: 当前噪声样本 x_t
        """
        prev_timestep = timestep - 1
        
        # 获取相关参数
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.ones_like(alpha_prod_t)
        beta_prod_t = 1 - alpha_prod_t
        
        # 计算 x_0 的预测值
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        # 计算均值 μ_θ(x_t, t)
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        # 用预测的x_0来计算采样均值
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        # 添加噪声（除了最后一step）
        if timestep > 0:
            noise = torch.randn_like(sample)
            variance = self.get_variance(timestep)
            pred_prev_sample = pred_prev_sample + (variance ** 0.5) * noise
            
        return pred_prev_sample

    def get_variance(self, timestep):
        """计算后验方差 σ_t^2"""
        prev_timestep = timestep - 1
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.ones_like(alpha_prod_t)
        
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[timestep]
        return variance
    
    def sample(self, model, shape, device='cpu', num_inference_steps=None):
        """
        完整的DDPM采样过程
    
        Args:
            model: 训练好的UNet模型
            shape: 生成样本的形状 (batch_size, channels, height, width)
            device: 设备
            num_inference_steps: 推理步数（默认使用训练时的全部步数）
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps
        
        # 从纯噪声开始
        sample = torch.randn(shape, device=device)
    
        # 设置时间步序列
        timesteps = torch.arange(num_inference_steps - 1, -1, -1, device=device)
    
        model.eval()
        with torch.no_grad():
            for t in timesteps:
                # 预测噪声
                timestep_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
                model_output = model(sample, timestep_batch)
            
                # 执行一步反向扩散
                sample = self.step(model_output, t, sample)
            
        return sample
    
    def training_loss(self, model, x_0, noise=None):
        """
        计算DDPM训练损失
    
        Args:
            model: UNet模型
            x_0: 原始清洁样本
            noise: 随机噪声（可选）
        """
        batch_size = x_0.shape[0]
        device = x_0.device
    
        # 随机采样时间步
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
    
        # 生成噪声
        if noise is None:
            noise = torch.randn_like(x_0)
    
        # 前向扩散加噪
        x_t = self.add_noise(x_0, noise, timesteps)
    
        # 模型预测噪声
        predicted_noise = model(x_t, timesteps)
    
        # 计算MSE损失
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
    
        return loss