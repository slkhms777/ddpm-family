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

for epoch in range(9, 2500, 10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DDPM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    visual_dir = '/mnt/data/gjx/Proj/ddpm-family/ddpm/pytorch/visualization/samples'
    checkpoint_dir = '/mnt/data/gjx/Proj/ddpm-family/ddpm/pytorch/checkpoints'

    model, optimizer = load_checkpoint(model, optimizer, epoch, checkpoint_dir, dataset='cifar10')
    generate_5x5_samples(model, epoch, device, visual_dir)
    print(f"Epoch {epoch + 1} samples generated and saved.")