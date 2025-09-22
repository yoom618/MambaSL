# ETSformer script generation
# hyperparameters setting are referenced from below:
# 1) [original] ETSformer source code (only for forecasting task)
#    : https://github.com/salesforce/ETSformer/tree/2566f689d022ec39acdcecd829898fcea17ba6dd/scripts
# 2) [tslib] time-series-library scripts
#    : https://github.com/thuml/Time-Series-Library/blob/85c08390b6ecc5a5c3bae33b3880b8bc3e413023/scripts/classification/ETSformer.sh

import os
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
    model = "ETSformer"

    dir_setting = {
        "data_dir" : "/data/user/MambaSL/dataset",
        "checkpoints": "/data/user/MambaSL/checkpoints",
    }

    data_configs = OmegaConf.load(f"{script_dir}/{data_metainfo}")

    gpu_setting = {
        "use_gpu" : True,
        "gpu_type" : "cuda",
        "gpu" : 1,
    }
    # gpu_setting = {
    #     "use_multi_gpu" : True,
    #     "devices" : "1,2"
    # }


    model_configs = {
        "e_layers" : [1,2],   # default: 2(original), 3(tslib)
        "n_heads" : [8],   # default: 8(both)
        "d_model" : [32,64,128,256,512],   # default: 512(original), 128(tslib)
        "d_ff" : [64,128,256,512,1024,2048],  # default: 2048(original), 256(tslib)
        "top_k" : [1,2,3,4],  # default: 3(both)
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
            replace_dict["batch_size"] = 4  # some hyperparamter settings require lower batch size
            replace_dict["gpu"] = 3
        if data_cfg["dataset"] == "EthanolConcentration":  # 9764MiB in batch_size=16
            replace_dict["gpu"] = 2
        if data_cfg["dataset"] == "MotorImagery":
            replace_dict["batch_size"] = 8  # GPU Memory Usage: 7394MiB
            replace_dict["gpu"] = 0
        if data_cfg["dataset"] == "StandWalkJump":
            replace_dict["batch_size"] = 8  # GPU Memory Usage: 6226MiB
        
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
            script_cfg["d_layers"] = script_cfg["e_layers"]
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