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