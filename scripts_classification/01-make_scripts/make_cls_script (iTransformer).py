# iTransformer script generation
# hyperparameters setting are referenced from below:
# 1) [original] iTransformer scripts
#    : https://github.com/thuml/iTransformer/tree/main/scripts/multivariate_forecasting
# 2) [tslib] time-series-library scripts (same lab)
#    : https://github.com/thuml/Time-Series-Library/blob/85c08390b6ecc5a5c3bae33b3880b8bc3e413023/scripts/classification/iTransformer.sh


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
    model = "iTransformer"

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
        "e_layers" : [1,2,3,4],   # default: 2,3,4(original), 2,3,4(tslib)
        "d_model" : [64,128,256,512,1024,2048],   # default: 128,256,512,1024(original), 128,512,2048(tslib)
        "d_ff" : [64,128,256,512,1024],  # default: 128,256,512,1024(original), 128,256,512(tslib)
        # "n_heads" : [8],  # default: 8(original), 8(tslib)
        "factor" : [1,3],  # default: 1(original), 1,3(tslib)
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
        if data_cfg["dataset"][0].lower() < "f":
            replace_dict["gpu"] = 0
        elif data_cfg["dataset"][0].lower() < "n":
            replace_dict["gpu"] = 1
        else:
            replace_dict["gpu"] = 2
        if data_cfg["dataset"] == "EigenWorms":
            replace_dict["gpu"] = 3
        if data_cfg["dataset"] == "DuckDuckGeese":
            replace_dict["batch_size"] = 8  # GPU Memory Usage: 9812MiB
            replace_dict["gpu"] = 3
        if data_cfg["dataset"] == "PEMS-SF":
            replace_dict["batch_size"] = 8  # GPU Memory Usage: 6402MiB
        
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

        with open(script_path.format(model, data_key), "w") as f:
            f.write(scripts)