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
import torch
import torch.optim as optim
import torch.nn.utils as utils
from torch.optim.lr_scheduler import LambdaLR

from models.DDPM import DDPM
from utils.dataset import get_cifar10_dataloader
from utils.checkpoint import save_checkpoint, load_checkpoint, save_ema_model, load_ema_model
from utils.visual import generate_5x5_samples
import numpy as np
from tqdm import tqdm

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """余弦退火学习率调度"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

# 简化的EMA类
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                else:
                    self.shadow[name] = param.data.clone()
                    
    def copy_to(self, target_model):
        """将EMA权重复制到目标模型"""
        for name, param in target_model.named_parameters():
            if name in self.shadow and param.requires_grad:
                param.data = self.shadow[name].clone()
                
    def save_state_dict(self):
        """返回EMA权重的state_dict"""
        ema_state_dict = {}
        for name, param in self.model.named_parameters():
            if name in self.shadow and param.requires_grad:
                ema_state_dict[name] = self.shadow[name].clone()
        return ema_state_dict

def train_ddpm(dataloader, num_epochs, checkpoint_dir, visual_dir):
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model_without_dp = DDPM()
    
    # 使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model_without_dp)
    else:
        model = model_without_dp
    
    model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    # 学习率调度器
    total_steps = num_epochs * len(dataloader)
    warmup_steps = total_steps // 10  # 10%预热
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # EMA初始化 - 始终使用原始模型（非DataParallel）
    ema = EMA(model_without_dp)
    ema.register()

    # 保存损失 
    losses = []
    step_count = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            # 训练步骤
            optimizer.zero_grad()
            
            # 前向传播
            if isinstance(model, torch.nn.DataParallel):
                loss = model.module.training_step(data)
            else:
                loss = model.training_step(data)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if isinstance(model, torch.nn.DataParallel):
                utils.clip_grad_norm_(model.module.parameters(), max_norm=1.0)
            else:
                utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # 每步更新学习率
            
            # 更新EMA
            ema.update()
            
            epoch_loss += loss.item()
            step_count += 1

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
        # 保存检查点、可视化
        if (epoch + 1) % 10 == 0:
            print(f"Saving checkpoint at epoch {epoch+1}")
            
            # 保存训练状态
            save_checkpoint(model_without_dp, optimizer, epoch, checkpoint_dir, ema=ema)
            
            # 创建临时模型用于EMA推理
            temp_model = DDPM().to(device)
            temp_model.load_state_dict(model_without_dp.state_dict())
            
            # 应用EMA权重到临时模型
            ema.copy_to(temp_model)
            
            # 使用EMA权重生成样本
            save_ema_model(temp_model, None, epoch, checkpoint_dir)
            generate_5x5_samples(temp_model, epoch, device, visual_dir)
            
            # 删除临时模型释放内存
            del temp_model
            torch.cuda.empty_cache()
            
            print(f"Checkpoint and samples saved for epoch {epoch+1}")
        
    return losses
