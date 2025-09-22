# GPT4TS script generation
# hyperparameters setting are referenced from below:
# 1) Original One-fits-all paper and scripts
#    : https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All/tree/9f3875dbba9f52ee534ac0902572502746f3b29d/Classification/scripts


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
    model = "GPT4TS"

    dir_setting = {
        "data_dir" : "/data/user/MambaSL/dataset",
        "checkpoints": "/data/user/MambaSL/checkpoints",
        "huggingface_cache_dir": "/data/user/MambaSL/huggingface/",
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
        "e_layers" : list(range(3, 7)),  # default: 3,6
        
        ### this should be fixed to 768 since we use pre-trained GPT2 model
        "d_model" : [768],  # default: 768

        ### d_ff is actually not used in classification task. it is used to trim the final output of transformer in other tasks
        "d_ff" : [768],  # default: 768
        
        ### while patch size from 4 to 32 were used in the original paper for 10 UEA datasets
        ### we set from 1/25 ~ 1/5 of seq_len in UEA datasets (where seq_len varies from 8 to 17984)
        ### And we set stride as {1/4, 1/2, or same size} of patch_size (as in the original paper)
        ### this is quite differ from PatchTST, since we test more various patch sizes and strides to match the number of hyperparameter combinations
        # "patch_size" : [16],  # default: 4, 8, 16, 32
        # "patch_stride" : [4,8,16],  # default: 1, 2, 4, 8, 16
        "patch_size_ratio" : [1/i for i in range(5, 26)],  # default: about 1/20 ~ 1/4 of seq_len except for some long seq_len datasets
        
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
    os.makedirs(dir_setting["huggingface_cache_dir"], exist_ok=True)

    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        replace_dict = {}
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]
        
        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]
        data_cfg["huggingface_cache_dir"] = dir_setting["huggingface_cache_dir"]
        
        model_cfg_tmp = model_configs.copy()
        model_cfg_tmp["patch_size & stride"] = []
        for ps_ratio in sorted(model_configs["patch_size_ratio"], reverse=True):
            ps = math.ceil(data_cfg["seq_len"] * ps_ratio)
            if ps in [0,1] or ps in set([i for i, _ in model_cfg_tmp["patch_size & stride"]]):
                continue
            for stride in sorted(set([math.ceil(data_cfg["seq_len"] * ps_ratio * stride_ratio) for stride_ratio in [1, 1/2, 1/4]]), reverse=True):
                if stride == 0:
                    continue
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