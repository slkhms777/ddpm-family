import torch


def save_checkpoint(model, optimizer, epoch, checkpoint_path, dataset='cifar10'):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    if dataset == 'cifar10':
        checkpoint_path = f"{checkpoint_path}/cifar10_epoch_{epoch+1}.pth"
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, epoch, scheduler, checkpoint_path, dataset='cifar10'):
    """加载模型检查点"""
    if dataset == 'cifar10':
        checkpoint_path = f"{checkpoint_path}/cifar10_epoch_{epoch+1}.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, scheduler

