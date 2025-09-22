# FEDformer script generation
# hyperparameters setting are referenced from below:
# 1) [original] FEDformer source code (only for forecasting task)
#    : https://github.com/MAZiqing/FEDformer/tree/c0f6b972def125691434d62be1ecadf710ae921a/scripts
# 2) [tslib] time-series-library scripts
#    : https://github.com/thuml/Time-Series-Library/blob/85c08390b6ecc5a5c3bae33b3880b8bc3e413023/scripts/classification/FEDformer.sh


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
    model = "FEDformer"

    dir_setting = {
        "data_dir" : "/data/yoom618/TSLib/dataset",
        "checkpoints": "/data/yoom618/TSLib/checkpoints",
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
        ### instead of moving_avg, moving_avg_ratio will be used for variable kernel size since sequence length is different
        ### e.g. 1 means 1% of sequence length
        # "moving_avg" : [25],  # default: 25
        "moving_avg_ratio" : [0.5, 1, 2, 3, 5, 10, 20, 30, 40, 50],  
        
        "e_layers" : [2],  # default: 2(original), 3(tslib)
        "n_heads" : [8],   # default: 8(both)
        "d_model" : [32,64,128,256,512],   # default: 512(original), 128(tslib)
        "d_ff" : [128,256,512,1024,2048],  # default: 2048(original), 256(tslib)
    }

    training_configs = {
        "is_training" : 1,
        "batch_size" : 16,
        "des" : "Exp",
        "itr" : 1,
        "dropout" : 0.1,
        "learning_rate" : 0.001,
        "train_epochs" : 100,
        "patience" : 10,
    }
    
    os.makedirs(f"{script_dir}/scripts_baseline", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)

    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        replace_dict = {}
        if data_cfg["dataset"] == "EigenWorms":
            replace_dict["batch_size"] = 4  # GPU Memory Use : 8568MiB
            replace_dict["gpu"] = 3
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]
        
        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]
        
        model_cfg_tmp = model_configs.copy()
        model_cfg_tmp["moving_avg"] = sorted(set([math.ceil(data_cfg["seq_len"] * (ratio/100)) for ratio in model_configs["moving_avg_ratio"]]))
        model_cfg_tmp.pop("moving_avg_ratio")
        model_configs_combination = reversed(make_combination(model_cfg_tmp))
        
        for model_cfg in model_configs_combination:
            script_cfg = gpu_setting.copy()
            script_cfg.update(data_cfg)
            script_cfg["model"] = model
            script_cfg["model_id"] = model_id.format(data_key)
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