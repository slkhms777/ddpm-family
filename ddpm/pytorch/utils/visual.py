import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import os

def generate_5x5_samples(model, epoch, device, visual_dir, dataset='cifar10'):
    """生成5x5的样本图像并保存"""
    # 如果模型是DataParallel实例，则使用model.module进行采样
    model_to_sample = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    model_to_sample.eval()
    with torch.no_grad():
        # 生成5x5的样本
        shape = (5 * 5, 3, 32, 32)
        samples = model_to_sample.sample(shape=shape, device=device)
        
        # 转换为numpy数组
        samples = samples.cpu().numpy()

        # 将样本从[-1, 1]范围转换为[0, 1]
        samples = samples * 0.5 + 0.5  
        
        # 确保在有效范围内
        samples = np.clip(samples, 0, 1)
        
        # 转换为[0, 255]范围的uint8
        samples = (samples * 255).astype(np.uint8)
        
        # 调整维度顺序：从(batch, channel, height, width)到(batch, height, width, channel)
        samples = samples.transpose(0, 2, 3, 1)  # BCHW -> BHWC
        
        # 创建一个5x5的网格图像
        grid_image = np.zeros((32 * 5, 32 * 5, 3), dtype=np.uint8)
        
        for i in range(5):
            for j in range(5):
                idx = i * 5 + j
                grid_image[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = samples[idx]
        
        # OpenCV使用BGR格式，需要转换
        grid_image = cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR)

        if dataset == 'cifar10':
            visual_dir = os.path.join(visual_dir, 'cifar10_samples')

        # 确保输出目录存在
        os.makedirs(visual_dir, exist_ok=True)

        # 保存图像
        filename = f"{visual_dir}/epoch_{epoch+1}_samples.png"
        cv2.imwrite(filename, grid_image)

def draw_loss_curve(losses, visual_dir, dataset='cifar10'):
    """绘制损失曲线并保存"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    if dataset == 'cifar10':
        visual_dir = os.path.join(visual_dir, 'cifar10_loss_curve')

    # 确保输出目录存在
    os.makedirs(visual_dir, exist_ok=True)

    # 保存图像
    filename = f"{visual_dir}/loss_curve.png"
    plt.savefig(filename)
    plt.close()
