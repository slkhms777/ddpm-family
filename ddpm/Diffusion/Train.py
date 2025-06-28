
import os
from typing import Dict
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler
from utils.visual import show_Loss_and_lr, generate_samples
from utils.FIDIS import FID_and_IS
def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root='../data/CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["ckpt_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    losses = []
    lrs = []
    
    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            epochLoss = 0
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                epochLoss += loss.item()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
            losses.append(epochLoss / len(dataloader))
        lrs.append(optimizer.state_dict()['param_groups'][0]["lr"])
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["ckpt_dir"], 'ckpt_' + str(e) + "_.pt"))
        
    os.makedirs(modelConfig["visual_dir"], exist_ok=True)
    # 保存loss为CSV
    loss_df = pd.DataFrame({
        'loss': losses
    })
    loss_csv_path = os.path.join(modelConfig["visual_dir"], 'ddpm_losses.csv')
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"Training losses saved to {loss_csv_path}")
    # 保存lr为CSV
    lr_df = pd.DataFrame({
        'lr': lrs
    })
    lr_csv_path = os.path.join(modelConfig["visual_dir"], 'ddpm_lrs.csv')
    lr_df.to_csv(lr_csv_path, index=False)
    print(f"Learning rates saved to {lr_csv_path}")

def eval(modelConfig: Dict):
    with torch.no_grad():
        # 采样最终图片以及不同步长的图片
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["ckpt_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
    
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # generate_samples(sampler, device=device, modelConfig=modelConfig)

        # 绘制loss曲线，lr曲线，以及累积时间
        show_Loss_and_lr(
            losses=pd.read_csv(os.path.join(modelConfig["visual_dir"], 'ddpm_losses.csv')),
            lrs=pd.read_csv(os.path.join(modelConfig["visual_dir"], 'ddpm_lrs.csv')),
            visual_dir=modelConfig["visual_dir"],
            model_name="ddpm"
        )


        # 计算FID和IS
        # 在eval函数中使用
        calculator = FID_and_IS(device="cuda", real_batch_size=10000, tmp_dir=modelConfig["tmp_dir"], is_splits=10)

        # 生成假图片
        calculator.prepare_fake_images(
            sampler=sampler,
            num_images=10000,
            batch_size=100,
            img_size=32,
            device=device
        )

        # 同时计算FID和IS
        results = calculator.compute_both()
        print(f"FID: {results['fid']:.4f}")
        print(f"IS: {results['is_mean']:.4f} ± {results['is_std']:.4f}")
        
        os.makedirs(modelConfig["visual_dir"], exist_ok=True)
        with open(os.path.join(modelConfig["visual_dir"], 'fid_is_results.txt'), 'w') as f:
            f.write(f"FID: {results['fid']:.4f}\n")
            f.write(f"IS: {results['is_mean']:.4f} ± {results['is_std']:.4f}\n")

    # 不同样本对比

    