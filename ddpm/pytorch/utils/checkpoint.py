import torch


def save_checkpoint(model, optimizer, epoch, checkpoint_path, ema=None, dataset='cifar10'):
    """保存模型检查点，包括EMA权重"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    # 保存EMA权重
    if ema is not None:
        checkpoint['ema_shadow'] = ema.shadow.copy()
    
    if dataset == 'cifar10':
        checkpoint_path = f"{checkpoint_path}/cifar10_epoch_{epoch+1}.pth"
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, epoch, checkpoint_path, ema=None, dataset='cifar10'):
    """加载模型检查点，包括EMA权重"""
    if dataset == 'cifar10':
        checkpoint_path = f"{checkpoint_path}/cifar10_epoch_{epoch+1}.pth"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载EMA权重
    if ema is not None and 'ema_shadow' in checkpoint:
        ema.shadow = checkpoint['ema_shadow']
    
    return model, optimizer

def save_ema_model(model, ema, epoch, checkpoint_path, dataset='cifar10'):
    """保存已经应用EMA权重的模型"""
    ema_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    if dataset == 'cifar10':
        ema_path = f"{checkpoint_path}/cifar10_ema_epoch_{epoch+1}.pth"
    torch.save(ema_checkpoint, ema_path)

def load_ema_model(model, epoch, checkpoint_path, dataset='cifar10'):
    """加载EMA模型用于推理"""
    if dataset == 'cifar10':
        ema_path = f"{checkpoint_path}/cifar10_ema_epoch_{epoch+1}.pth"
    checkpoint = torch.load(ema_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

