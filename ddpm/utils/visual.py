import os
import matplotlib.pyplot as plt

def showLoss(losses, visual_dir=None, model_name=None):
    if visual_dir is None:
        print("No visual dir provided, skipping loss visualization.")
        return

    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    plt.figure()
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve - {model_name}' if model_name else 'Training Loss Curve')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(visual_dir, f'loss_curve_{model_name}.png' if model_name else 'loss_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")


