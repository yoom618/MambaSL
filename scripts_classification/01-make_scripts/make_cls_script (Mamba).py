# Mamba model given in tslib
# made hyperparameter settings by referring to forcasting scripts

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
    script_path = f"{script_dir}/scripts_baseline/Mamba_{{}}.sh" ###############
    model_id = "{}"
    model = "Mamba"

    dir_setting = {
        "data_dir" : "/data/yoom618/TSLib/dataset",
        "checkpoints": "/data/yoom618/TSLib/checkpoints",
    }

    data_configs = OmegaConf.load(f"{script_dir}/{data_metainfo}")

    gpu_setting = {
        "use_gpu" : True,
        "gpu_type" : "cuda",
        "gpu" : 0
    }
    # gpu_setting = {
    #     "use_multi_gpu" : True,
    #     "devices" : "1,2"
    # }


    
    model_configs = {
        "e_layers" : [1,2,3,4], # default : 2. use 1 layer to compare the basic performance of Mamba
        "expand" : [1,2],  # default : 2
        "d_ff" : [1, 2, 4, 8, 16], # default : 16
        "d_model" : [32, 64, 128, 256, 512, 1024], # default : 128
        "d_conv" : [4],  # default : 4. only <=4 is available due to causal conv layer
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

    os.makedirs(f"{script_dir}/scripts_mamba", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)
    
    for data_key, data_cfg in data_configs.items():
        scripts = ""
        

        replace_dict = {}
        if data_cfg["dataset"] == "EigenWorms":
            replace_dict["batch_size"] = 4  # GPU Memory Usage: 9136MiB
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]

        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]

        model_configs_combination = reversed(make_combination(model_configs))
        
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

        with open(script_path.format(data_key), "w") as f:
            f.write(scripts)