# PatchTST script generation
# hyperparameters setting are referenced from below:
# 1) [original] PatchTST source code (only for forecasting task)
#    : https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9/PatchTST_supervised/scripts/PatchTST
# 2) [tslib] time-series-library scripts
#    : https://github.com/thuml/Time-Series-Library/blob/85c08390b6ecc5a5c3bae33b3880b8bc3e413023/scripts/classification/PatchTST.sh



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
    model = "PatchTST"

    dir_setting = {
        "data_dir" : "/data/yoom618/TSLib/dataset",
        "checkpoints": "/data/yoom618/TSLib/checkpoints",
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
        "e_layers" : [1, 2, 3],  # default: 3(both)
        
        ### n_heads=4 for small d_model, n_heads=16 for large d_model in original paper
        ### thus, n_heads=4 for d_model=16,32 & n_heads=16 for d_model=64,128 in this experiment
        ### this will be automatically set in the script
        # "n_heads" : [4, 16],   # default: 4,16(original), 8(tslib)

        "d_model" : [16, 32, 64, 128],  # default: 16,128(original), 128(tslib)
        
        "d_ff" : [64, 128, 256],  # default: 128,256(original), 256(tslib)

        ### since patch size in 8~16 seemed to be the best choice for seq_len 336 in original paper,
        ### we set from 2.5% ~ 25% of seq_len in UEA datasets (where seq_len varies from 8 to 17984)
        ### And we set stride as 1/2 of the patch size (same as original paper)
        # "patch_size" : [16],  # default: 16(both)
        # "patch_stride" : [8],  # default: 8(both)
        "patch_size_ratio" : [2.5, 5, 7.5, 10, 15, 20, 25],  # patch size is inversely proportional to computational cost
        
    }

    training_configs = {
        "is_training" : 1,
        "batch_size" : 16,
        "des" : "Exp",
        "itr" : 1,
        "dropout" : 0.1,  # default: 0.2,0.3(original), 0.1(tslib)
        "learning_rate" : 0.001,
        "train_epochs" : 100,
        "patience" : 10,
    }
    
    os.makedirs(f"{script_dir}/scripts_baseline", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)

    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        replace_dict = {}
        if data_cfg["dataset"] == "DuckDuckGeese":
            replace_dict["batch_size"] = 4  # 10376MiB for patch_size=14, patch_stride=7. lower patch size requires lower batch size
            replace_dict["gpu"] = 2
        if data_cfg["dataset"] == "PEMS-SF":
            replace_dict["batch_size"] = 4  # 9210MiB for patch_stride=2
            replace_dict["gpu"] = 2
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"]
        
        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]
        
        model_cfg_tmp = model_configs.copy()
        model_cfg_tmp["patch_size & stride"] = []
        for ps_ratio in sorted(model_configs["patch_size_ratio"], reverse=True):
            ps = math.ceil(data_cfg["seq_len"] * (ps_ratio / 100))
            stride = math.ceil(data_cfg["seq_len"] * (ps_ratio / 100) * 1/2)
            if ps in [0,1] or ps in set([i for i, _ in model_cfg_tmp["patch_size & stride"]]):
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
            model_cfg["n_heads"] = 4 if model_cfg["d_model"] in [16, 32] else 16
            script_cfg.update(model_cfg)
            script_cfg.update(training_configs)
            script_cfg.update(replace_dict)
            if data_cfg["dataset"] == "DuckDuckGeese" and model_cfg["patch_stride"] <= 4:
                script_cfg["batch_size"] = 2
            if data_cfg["dataset"] == "PEMS-SF" and model_cfg["patch_stride"] <= 2:
                script_cfg["batch_size"] = 2
            del script_cfg["dataset"]

            script = f"python run.py \\\n"
            for key, value in script_cfg.items():
                script += f"  --{key} {value} \\\n"
            script = script[:-3]
            script += "\n\n"
            scripts += script
            print(script)

        with open(script_path.format(model, data_key), "w") as f:
            f.write(scripts)