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

from models.DDPM import DDPM
from utils.dataset import get_cifar10_dataloader
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.visual import generate_5x5_samples
import numpy as np
from tqdm import tqdm

def train_ddpm(dataloader, num_epochs, checkpoint_dir, visual_dir):
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model_without_dp = DDPM() # 保存原始模型的引用
    
    # 使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model_without_dp)
    else:
        model = model_without_dp
    
    model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    # 保存损失 
    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            
            # 训练步骤
            optimizer.zero_grad()
            # 如果使用DataParallel，则通过model.module访问自定义方法
            if isinstance(model, torch.nn.DataParallel):
                loss = model.module.training_step(data)
            else:
                loss = model.training_step(data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        # 保存检查点、可视化
        if (epoch + 1) % 10 == 0:
            # 保存检查点时，通常保存原始模型的状态字典
            if isinstance(model, torch.nn.DataParallel):
                save_checkpoint(model.module, optimizer, epoch, checkpoint_dir)
            else:
                save_checkpoint(model, optimizer, epoch, checkpoint_dir)
            generate_5x5_samples(model, epoch, device, visual_dir) # generate_5x5_samples内部会处理DataParallel
        
    return losses
