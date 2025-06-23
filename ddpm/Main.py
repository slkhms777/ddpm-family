from Diffusion.Train import train, eval


def main(model_config = None):
    modelConfig = {
        "state": "eval",

        "epoch": 10,
        "batch_size": 64,
        "nrow": 8,

        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,

        "img_size": 32,
        "grad_clip": 1.,

        "device": "cuda:0", 

        "training_load_weight": None,
        "test_load_weight": "ckpt_199_.pt",

        "ckpt_dir": "./Checkpoints/",
        "sampled_dir": "./SampledImgs/ddpm",
        "visual_dir": "./Visualization/ddpm",
        "tmp_dir": "./tmp_fid/ddpm",
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
    main()