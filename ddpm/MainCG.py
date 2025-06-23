from DiffusionCG.TrainCG import train, eval
from utils.visual import showLoss

def main(model_config=None):
    modelConfig = {
        "state": "train", # or eval
        "epoch": 70,
        "batch_size": 80,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "s": 1.8,
        "training_load_weight": None,
        "test_load_weight": "ckpt_63_.pt",

        "ckpt_dir": "./CheckpointsCG/",
        "sampled_dir": "./SampledImgs/ddpmcg",
        "visual_dir": "./Visualization/ddpmcg",
        "tmp_dir": "./tmp_fid/ddpmcg",
        "sampledNoisyImgName": "NoisyImgs",
        "sampledImgName": "SampledImgs",
        "nrow": 8,
        "classifier_weight_path": "./cifar10-classifier/checkpoint/ckpt.pth"
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        losses = train(modelConfig)
        showLoss(losses, visual_dir=modelConfig["visual_dir"])
    else:
        eval(modelConfig)


if __name__ == '__main__':
    main()