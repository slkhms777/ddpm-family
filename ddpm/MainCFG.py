from DiffusionCFG.TrainCFG import train, eval

def main(model_config=None):
    modelConfig = {
        "state": "eval", # or eval
        "epoch": 10,
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
        "device": "cuda:2",
        "w": 1.8,
        "training_load_weight": None,
        "test_load_weight": "ckpt_63_.pt",

        "ckpt_dir": "./CheckpointsCFG/",
        "sampled_dir": "./SampledImgs/ddpmcfg",
        "visual_dir": "./Visualization/ddpmcfg",
        "tmp_dir": "./tmp_fid/ddpmcfg",
        "sampledNoisyImgName": "NoisyImgs",
        "sampledImgName": "SampledImgs",
        "nrow": 8
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    modelConfig = {
        "state": "train", 
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
        "device": "cuda:1",
        "w": 1.8,
        "training_load_weight": None,
        "test_load_weight": "ckpt_63_.pt",

        "ckpt_dir": "./CheckpointsCFG/",
        "sampled_dir": "./SampledImgs/ddpmcfg",
        "visual_dir": "./Visualization/ddpmcfg",
        "tmp_dir": "./tmp_fid/ddpmcfg",
        "sampledNoisyImgName": "NoisyImgs",
        "sampledImgName": "SampledImgs",
        "nrow": 8
    }
    # main(model_config=modelConfig)
    modelConfig["state"] = "eval"
    main(model_config=modelConfig)