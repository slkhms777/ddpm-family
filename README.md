# DDPM Family

## 快速开始

以DDPM为例：

**训练：**
```bash
python Main.py
```

**采样与评估：**
```bash
# 修改Main.py中的modelConfig["state"] = "eval"
python Main.py
```

## 目录结构

```
ddpm/
├── Main.py                # DDPM主入口
├── MainCFG.py             # Classifier-Free Guidance主入口
├── MainCG.py              # Classifier Guidance主入口
├── Scheduler.py           # 学习率调度器
├── Checkpoints/           # 权重保存目录
├── CheckpointsCFG/
├── CheckpointsCG/
├── cifar10_classifier/    # 分类器相关代码
│   ├── DLA.py
│   ├── main.py
│   └── utils.py
├── Diffusion/             # DDPM核心代码
│   ├── __init__.py
│   ├── Diffusion.py
│   ├── Model.py
│   └── Train.py
├── DiffusionCFG/          # CFG相关代码
│   ├── __init__.py
│   ├── DiffusionCFG.py
│   ├── ModelCFG.py
│   └── TrainCFG.py
├── DiffusionCG/           # CG相关代码
│   ├── __init__.py
│   ├── DiffusionCG.py
│   ├── ModelCG.py
│   └── TrainCG.py
├── utils/
│   ├── FIDIS.py           # FID/IS评估工具
│   └── visual.py          # 可视化工具
```