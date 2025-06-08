
from utils.dataset import get_cifar10_dataloader
from train.train_cifar10 import train_ddpm
from utils.visual import draw_loss_curve
import os
data_dir = '/mnt/data/gjx/Proj/ddpm-family/data'
visual_dir = '/mnt/data/gjx/Proj/ddpm-family/pytorch/visualization'
checkpoint_dir = '/mnt/data/gjx/Proj/ddpm-family/pytorch/checkpoints'
os.makedirs(visual_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)  
os.makedirs(data_dir, exist_ok=True)  # 确保数据目录存在
def main():
    batch_size = 128
    num_epochs = 2500
    dataloader = get_cifar10_dataloader(batch_size=batch_size, root_dir=data_dir)
    losses = train_ddpm(dataloader=dataloader, num_epochs=num_epochs, checkpoint_dir=checkpoint_dir, visual_dir=visual_dir)
    draw_loss_curve(losses, visual_dir=visual_dir, dataset='cifar10')


if __name__ == '__main__':
    main()