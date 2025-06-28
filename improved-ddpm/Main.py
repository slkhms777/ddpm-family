from Diffusion.Train import train, eval

def main(model_config=None):
    modelConfig = {
        "state": "train",
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "attn": [1],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "test_load_weight": "ckpt_199_.pt",
        "ckpt_dir": "./Checkpoints/",
        "sampled_dir": "./SampledImgs/improved_ddpm",
        "visual_dir": "./Visualization/improved_ddpm",
        "tmp_dir": "./tmp_fid/improved_ddpm",
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
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "attn": [1],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "test_load_weight": "ckpt_199_.pt",
        "ckpt_dir": "./Checkpoints/",
        "sampled_dir": "./SampledImgs/improved_ddpm",
        "visual_dir": "./Visualization/improved_ddpm",
        "tmp_dir": "./tmp_fid/improved_ddpm",
        "sampledNoisyImgName": "NoisyImgs",
        "sampledImgName": "SampledImgs",
        "nrow": 8
    }
    # main(modelConfig)
    modelConfig["state"] = "eval"
    main(modelConfig)
