import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

def show_Loss_and_lr(losses, lrs, visual_dir=None, model_name=None):
    if visual_dir is None:
        print("No visual dir provided, skipping loss/lr visualization.")
        return

    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.plot(losses, label='Loss', color='tab:blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    plt.title(f'Training Loss & LR Curve - {model_name}' if model_name else 'Training Loss & LR Curve')
    plt.grid(True)

    if lrs is not None:
        ax2 = ax1.twinx()
        ax2.plot(lrs, label='LR', color='tab:orange')
        ax2.set_ylabel('Learning Rate', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

    lines, labels = ax1.get_legend_handles_labels()
    if lrs is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    plt.legend(lines, labels, loc='best')

    save_path = os.path.join(visual_dir, f'loss_lr_curve_{model_name}.png' if model_name else 'loss_lr_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Loss & LR curve saved to {save_path}")



def generate_samples(sampler, device='cpu', modelConfig=None):
    sampler.eval()
    with torch.no_grad():
        # 采样图片
        noisyImage = torch.randn(size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]+".png"), nrow=modelConfig["nrow"])
        
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]+".png"), nrow=modelConfig["nrow"])

        # 采样不同步长的图片
        noisyImage = torch.randn(size=[10, 3, 32, 32], device=device)
        sampledImgs = torch.stack(sampler(noisyImage, multi_steps=True), dim=0)  # [10, 11, 3, 32, 32]
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # 999 899 799 ... 0
        for i, img in enumerate(sampledImgs):
            save_image(img, os.path.join(
                modelConfig["sampled_dir"],  modelConfig["sampledImgName"]+f"_{i*100}.png"), nrow=10)
            
def generate_samples_by_classes(sampler, device='cpu', modelConfig=None):
    sampler.eval()
    with torch.no_grad():
        # 生成标签
        step = int(modelConfig["batch_size"] // 10)
        labelList = []
        k = 0
        for i in range(1, modelConfig["batch_size"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        print("labels: ", labels)

        # 采样图片
        noisyImage = torch.randn(size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]+".png"), nrow=modelConfig["nrow"])
        
        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]+".png"), nrow=modelConfig["nrow"])


        labels = torch.arange(10).long().to(device) + 1
        # 采样不同步长的图片
        noisyImage = torch.randn(size=[10, 3, 32, 32], device=device)
        sampledImgs = torch.stack(sampler(noisyImage, labels, multi_steps=True), dim=0)  # [10, 11, 3, 32, 32]
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # 999 899 799 ... 0
        for i, img in enumerate(sampledImgs):
            save_image(img, os.path.join(
                modelConfig["sampled_dir"],  modelConfig["sampledImgName"]+f"_{i*100}.png"), nrow=10)  
