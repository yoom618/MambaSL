# ModernTCN script generation
# hyperparameters setting are referenced from below:
# 1) Original ModernTCN paper and scripts
#    : https://github.com/luodhhh/ModernTCN/blob/56a9a2c018385cd5acef015378cae7f084d1b11c/ModernTCN-classification/scripts/classification.sh


import os
import math
from omegaconf import OmegaConf
import itertools

def make_combination(config_dict):
    keys, values = zip(*config_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


if __name__ == "__main__":
    script_dir = "./scripts_classification"
    data_metainfo = "data_classification.yaml"
    script_path = f"{script_dir}/scripts_baseline/{{}}_{{}}.sh"
    model_id = "{}"
    model = "ModernTCN"

    dir_setting = {
        "data_dir" : "/data/user/MambaSL/dataset",
        "checkpoints": "/data/user/MambaSL/checkpoints",
    }

    data_configs = OmegaConf.load(f"{script_dir}/{data_metainfo}")

    gpu_setting = {
        "use_gpu" : True,
        "gpu_type" : "cuda",
        "gpu" : 0,
    }
    # gpu_setting = {
    #     "use_multi_gpu" : True,
    #     "devices" : "1,2"
    # }


    model_configs = {
        # too much hyperparameters to be tuned 
        # + unknown hyperparameter combinations that was tested by the original paper
        # thus we made 252 combinations based on the original paper's hyperparameters

        "ffn_ratio" : [1, 2, 4], # default: 1, 2, 4

        ### set patch_size using PatchTST paper's setting, since the original paper didn't mention about it
        # "patch_size" : [8], # default: 1, 8, 32, 48, 64
        # "patch_stride" : [4], # default: 1, 4, 16, 24, 32 (1/2 of patch_size except 1)
        "patch_size_ratio" : [2.5, 5, 7.5, 10, 15, 20, 25], 

        ### simplified than the original paper since it's difficult to find the hyperparameter setting rule in the original paper
        # (actually, the results are worse than the originals' in EthanolConcentration & FaceDetection though the settings are included in our experiment)
        # (in contrast, the results are better in other datasets which use the hyperparameter settings not included in our experiment)
        "num_blocks - large_size - small_size - dims" : [
            ("1", "13", "5", "32"), ("1", "13", "5", "64"), ("1", "13", "5", "128"), ("1", "13", "5", "256"),
            ("1 1", "9 9", "5 5", "32 64"), ("1 1", "9 9", "5 5", "64 128"), ("1 1", "9 9", "5 5", "128 256"),
            ("1 1", "13 13", "5 5", "32 64"), ("1 1", "13 13", "5 5", "64 128"), ("1 1", "13 13", "5 5", "128 256"),
            ("1 1 1", "9 9 9", "5 5 5", "32 64 128"), ("1 1 1", "13 13 13", "5 5 5", "32 64 128"),
        ],
        # default: 1        , 13        , 5         , 32        for SelfRegulationSCP1   (seq_len=896, enc_in=6)
        #          1        , 13        , 5         , 256       for Handwriting          (seq_len=152, enc_in=3)
        #          1        , 31        , 5         , 256       for Heartbeat            (seq_len=405, enc_in=61)
        #          2        , 91        , 5         , 32        for PEMS-SF              (seq_len=144, enc_in=963)
        #          1 1      , 13 13     , 5 5       , 32 64     for EthanolConcentration (seq_len=1751, enc_in=3)
        #          1 1      , 21 19     , 5 5       , 256 512   for JapaneseVowels       (seq_len=29, enc_in=12)
        #          1 1      , 51 49     , 5 5       , 64 128    for SelfRegulationSCP2   (seq_len=1152, enc_in=7)
        #          1 1      , 51 49     , 5 5       , 128 256   for UWaveGestureLibrary  (seq_len=315, enc_in=3)
        #          1 1 1    , 1 1 1     , 5 5 5     , 32 64 128 for SpokenArabicDigits   (seq_len=93, enc_in=13)
        #          1 1 1    , 9 9 9     , 5 5 5     , 32 64 128 for FaceDetection        (seq_len=62, enc_in=144)
        
    }

    training_configs = {
        "is_training" : 1,
        "batch_size" : 16,
        "des" : "Exp",
        "itr" : 1,
        "dropout" : 0.1,
        "learning_rate" : 0.001,
        "train_epochs" : 50,  # same as original paper
        "patience" : 10,
    }
    
    os.makedirs(f"{script_dir}/scripts_baseline", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)

    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        replace_dict = {}
        # DuckDuckGeese & PEMS-SF : cannot run training on 1080 (thus use A100 via Google Colab)
        # dims higher than 64 normaly requires more than 40GiB in batch_size 16
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]
        
        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]
        
        model_cfg_tmp = model_configs.copy()
        model_cfg_tmp["patch_size & stride"] = []
        for ps_ratio in sorted(model_configs["patch_size_ratio"], reverse=True):
            ps = math.ceil(data_cfg["seq_len"] * (ps_ratio / 100))
            stride = math.ceil(data_cfg["seq_len"] * (ps_ratio / 100) * 1/2)
            if ps in [0] or ps in set([i for i, _ in model_cfg_tmp["patch_size & stride"]]):
                continue
            else:
                model_cfg_tmp["patch_size & stride"].append((ps, stride))
        model_cfg_tmp.pop("patch_size_ratio")
        model_configs_combination = reversed(make_combination(model_cfg_tmp))
        
        for model_cfg in model_configs_combination:
            script_cfg = gpu_setting.copy()
            script_cfg.update(data_cfg)
            script_cfg["model"] = model
            script_cfg["model_id"] = model_id.format(data_key)
            model_cfg["patch_size"], model_cfg["patch_stride"] = model_cfg["patch_size & stride"]
            model_cfg.pop("patch_size & stride")
            model_cfg["num_blocks"], model_cfg["large_size"], model_cfg["small_size"], model_cfg["dims"] = model_cfg["num_blocks - large_size - small_size - dims"]
            model_cfg.pop("num_blocks - large_size - small_size - dims")
            script_cfg.update(model_cfg)
            script_cfg.update(training_configs)
            script_cfg.update(replace_dict)

            script = f"python run.py \\\n"
            for key, value in script_cfg.items():
                script += f"  --{key} {value} \\\n"
            script = script[:-3]
            script += "\n\n"
            scripts += script
            print(script)

        with open(script_path.format(model, data_key), "w") as f:
            f.write(scripts)