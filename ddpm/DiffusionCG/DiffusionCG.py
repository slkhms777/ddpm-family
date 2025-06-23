import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, classifier, beta_1, beta_T, T, s=0.):
        super().__init__()

        self.model = model
        self.classifier = classifier
        self.T = T
        self.s = s

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / self.alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - self.alphas) / torch.sqrt(1. - self.alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, labels):
        # 方差计算保持不变
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        
        # 启用梯度计算（分类器需要梯度）
        with torch.enable_grad():
            x_t_grad = x_t.detach().requires_grad_(True)
            
            # 用分类器计算梯度引导
            classifier_output = self.classifier(x_t_grad, t)
            classifier_log_probs = F.log_softmax(classifier_output, dim=-1)
            
            # 选择对应标签的log概率
            selected_log_probs = classifier_log_probs[range(len(labels)), labels]
            
            # 计算梯度 ∇_x log p(y|x_t)
            classifier_grad = torch.autograd.grad(
                outputs=selected_log_probs.sum(),
                inputs=x_t_grad,
                create_graph=False
            )[0]
        
        # 用扩散模型预测噪声（无条件）
        eps = self.model(x_t, t, labels)
        
        # Classifier Guidance
        # 公式：ε̃ = ε - √(1-ᾱₜ) * s * ∇_x log p(y|x_t)
        sqrt_one_minus_alphas_bar = extract(torch.sqrt(1. - self.alphas_bar), t, x_t.shape)
        eps_guided = eps - sqrt_one_minus_alphas_bar * self.s * classifier_grad
        
        # 计算均值
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps_guided)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        x_t = x_T
        self.classifier.eval()
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   