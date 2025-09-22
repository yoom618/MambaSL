# TSLANet script generation
# hyperparameters setting are referenced from below:
# 1) Original TSLANet source code
#    : https://github.com/emadeldeen24/TSLANet/blob/main/Classification/run_datasets.sh


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
    model = "TSLANet"

    dir_setting = {
        "data_dir" : "/data/yoom618/TSLib/dataset",
        "log_dir": "/data/yoom618/TSLib/logs (TSLANet)",
        "result_dir": "_run_TSLANet/results",
    }

    data_configs = OmegaConf.load(f"{script_dir}/{data_metainfo}")

    gpu_setting = {
        "gpu" : 3,
    }

    model_configs = {
        "depth" : [1, 2, 3],  # default: 1 ~ 3. 3 for UEA in the original paper
        "emb_dim" : [32, 64, 128, 256], # default: 32 ~ 256. 256 for UEA in the original paper
        "mlp_ratio" : [1, 2, 3],  # default: 3

        ### use similar setting with other models' experiments
        # "patch_size" : [8],  # default: 8, 32, 64. 8 for UEA in the original paper
        "patch_size_ratio" : [2.5, 5, 7.5, 10, 15, 20, 25],

        "masking_ratio" : [0.4],  # default: 0.4
        "ICB" : [True],  # default: True
        "ASB" : [True],  # default: True
        "adaptive_filter" : [True],  # default: True
        "load_from_pretrained" : [True],  # default: True
        
    }

    training_configs = {
        "save_path" : f'"{dir_setting["log_dir"]}"',
        "batch_size" : 16,
        "dropout" : 0.1,  # default was 0.15. but we set to 0.1 for fair comparison
        "train_lr" : 0.001,
        "pretrain_lr" : 0.001,
        "num_epochs" : 100,
        "pretrain_epoch" : 50,
        "patience" : 10, # the original paper does not use Early Stopping, but we use it for fair comparison with other models
        "seed": 2021, # default was 42. but we set to 2021 for the same seed setting with tslib
    }
    
    os.makedirs(f"{script_dir}/scripts_baseline", exist_ok=True)
    os.makedirs(dir_setting["log_dir"], exist_ok=True)
    os.makedirs(dir_setting["result_dir"], exist_ok=True)

    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        replace_dict = {}
        
        data_cfg_tmp = dict()
        data_cfg_tmp["data_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg_tmp["data_type"] = 'uea'
        data_cfg_tmp["data_name"] = data_cfg["dataset"]
        os.makedirs(f"{dir_setting['result_dir']}/{data_cfg["dataset"]}", exist_ok=True)
        
        model_cfg_tmp = model_configs.copy()
        model_cfg_tmp["patch_size"] = sorted(set([max(2, math.ceil(data_cfg["seq_len"] * (ps_ratio / 100))) 
                                       for ps_ratio in model_cfg_tmp["patch_size_ratio"]]), reverse=True)
        model_cfg_tmp.pop("patch_size_ratio")
        model_configs_combination = reversed(make_combination(model_cfg_tmp))
        
        for model_cfg in model_configs_combination:
            script_cfg = gpu_setting.copy()
            script_cfg.update(data_cfg_tmp)
            script_cfg["model_id"] = model_id.format(data_key)
            script_cfg.update(model_cfg)
            script_cfg.update(training_configs)
            script_cfg.update(replace_dict)

            script = f"python -u _run_TSLANet/TSLANet_classification.py \\\n"
            for key, value in script_cfg.items():
                script += f"  --{key} {value} \\\n"
            script = script[:-3]
            script += f" > {dir_setting['result_dir']}/{data_cfg["dataset"]}/{data_cfg["dataset"]}_depth{model_cfg['depth']}_dm{model_cfg['emb_dim']}_mlp{model_cfg['mlp_ratio']}_ps{model_cfg['patch_size']}.log 2>&1"
            script += "\n\n"
            scripts += script
            print(script)

        with open(script_path.format(model, data_key), "w") as f:
            f.write(scripts)