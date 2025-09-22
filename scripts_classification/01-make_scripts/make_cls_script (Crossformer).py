# Crossformer script generation
# hyperparameters setting are referenced from below:
# 1) [original] Crossformer script
#    : https://github.com/Thinklab-SJTU/Crossformer/tree/c10c8eadb153d1dd9798250967747ca3ebb81383/scripts
# 2) [tslib] time-series-library scripts
#    : https://github.com/thuml/Time-Series-Library/blob/85c08390b6ecc5a5c3bae33b3880b8bc3e413023/scripts/classification/Crossformer.sh


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
    model = "Crossformer"

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

    # prev setting (only considererd tslib, UNUSED)
    # model_configs = {
    #     "e_layers" : [1,2,3,4],   # default: 3(tslib)
    #     "d_model" : [32,64,128,256],   # default: 128(tslib)
    #     "d_ff" : [64,128,256,512],  # default: 256(tslib)
    #     "n_heads" : [8],  # default: 8(tslib)
    #     "factor" : [1,2,3,4],  # default: 3(tslib)
    #     # "seg_len" : unconsidered since it was fixed to 12 in original tslib code
    # }

    model_configs = {
        "e_layers" : [1,2,3],   # default: 2,3(original), 3(tslib)
        "d_model" : [32,64,128,256],   # default: 64,256(original), 128(tslib)
        "d_ff" : [64,128,256,512],  # default: 128,512(original), 256(tslib)
        "n_heads" : [4],  # default: 2,4(original), 8(tslib)
        "factor" : [3,10],  # default: 10(original), 3(tslib)
        "seg_len_cf" : [6,12,24],  # default: 6,12,24(original), 12(tslib)
    }

    training_configs = {
        "is_training" : 1,
        "batch_size" : 16,  # original: 32, tslib: 16. set to 16 for fair comparison
        "des" : "Exp",
        "itr" : 1,
        "dropout" : 0.1,
        "learning_rate" : 0.001,  # original: 5e-3 ~ 1e-5, tslib: 1e-3. set to 1e-3 for fair comparison
        "train_epochs" : 100,  # original: 20, tslib: 100. set to 100 for fair comparison
        "patience" : 10,  # original: 3, tslib: 10. set to 10 for fair comparison
    }
    
    os.makedirs(f"{script_dir}/scripts_baseline", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)

    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        replace_dict = {}
        if data_cfg["dataset"] == "DuckDuckGeese":
            replace_dict["batch_size"] = 4  # GPU Memory Usage: 7224MiB
            replace_dict["gpu"] = 3
        if data_cfg["dataset"] == "EigenWorms":
            replace_dict["batch_size"] = 4
            replace_dict["gpu"] = 3
        if data_cfg["dataset"] == "MotorImagery":
            replace_dict["batch_size"] = 4  # GPU Memory Usage: 6006MiB
            replace_dict["gpu"] = 3
        if data_cfg["dataset"] == "PEMS-SF":
            replace_dict["batch_size"] = 8  # GPU Memory Usage: 5852MiB
        
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
            if (data_key == "CLS_EigenWorms") and (model_cfg["seg_len_cf"] == 6):
                script_cfg["batch_size"] = 2

            script = f"python run.py \\\n"
            for key, value in script_cfg.items():
                script += f"  --{key} {value} \\\n"
            script = script[:-3]
            script += "\n\n"
            scripts += script
            print(script)

        with open(script_path.format(model, data_key), "w") as f:
            f.write(scripts)