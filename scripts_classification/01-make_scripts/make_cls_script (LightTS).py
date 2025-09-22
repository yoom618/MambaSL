# LightTS script generation
# hyperparameters setting are referenced from below:
# 1) [original] LightTS source code (only for forecasting task)
#    : https://www.dropbox.com/scl/fo/zctja07myvwfrv8byu4gn/AO4lILVlUneO3rr-1X_ppCM?rlkey=vntqwdnpdcp6d40z50ymhxy0q&e=1&dl=0
#      (this is the annonymous link of the original repo referenced in the paper)
# 2) time-series-library scripts
#    : https://github.com/thuml/Time-Series-Library/blob/85c08390b6ecc5a5c3bae33b3880b8bc3e413023/scripts/classification/LightTS.sh


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
    model = "LightTS"

    dir_setting = {
        "data_dir" : "/data/user/MambaSL/dataset",
        "checkpoints": "/data/user/MambaSL/checkpoints",
    }

    data_configs = OmegaConf.load(f"{script_dir}/{data_metainfo}")

    gpu_setting = {
        "use_gpu" : True,
        "gpu_type" : "cuda",
        "gpu" : 2,
    }
    # gpu_setting = {
    #     "use_multi_gpu" : True,
    #     "devices" : "1,2"
    # }


    model_configs = {
        "d_model" : [32, 64, 128, 256, 512],  # default : 64,128,256(original), 128(tslib)

        ### instead of chunk_size, num_chunks will be used since seq_len varies
        # "chunk_size" : [7, 14, 24, 40, 48, 72, 80], # default : 7,14,24,40,48,72,80(original), 24(tslib)
        "num_chunks" : range(2, 22),  
        
    }

    training_configs = {
        "is_training" : 1,
        "batch_size" : 16,  # default : 4,16,32,128(original), 16(tslib)
        "des" : "Exp",
        "itr" : 1,
        "dropout" : 0.1,
        "learning_rate" : 0.001,  # default : 1e-5 ~ 1e-3(original), 1e-3(tslib)
        "train_epochs" : 100,  # default : 15,150,200(original), 100(tslib)
        "patience" : 10,
    }
    
    os.makedirs(f"{script_dir}/scripts_baseline", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)

    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        replace_dict = {}
        if data_cfg["dataset"] == "EigenWorms":
            replace_dict["batch_size"] = 8  # 6471MiB
            replace_dict["gpu"] = 3
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]
        
        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]
        
        model_cfg_tmp = dict()
        model_cfg_tmp["d_model"] = model_configs["d_model"]
        model_cfg_tmp["chunk_size"] = sorted(list(set([math.ceil(data_cfg["seq_len"] / num_chunks) for num_chunks in model_configs["num_chunks"]])))
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