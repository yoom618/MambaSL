# Peri-midformer script generation
# hyperparameters setting are referenced from below:
# 1) [original] Peri-midformer script
#    : https://github.com/QiangWu-AI/Peri-midFormer/tree/main/scripts/classification


import os
from omegaconf import OmegaConf
import itertools
import math

def make_combination(config_dict):
    keys, values = zip(*config_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


if __name__ == "__main__":
    script_dir = "./scripts_classification"
    data_metainfo = "data_classification.yaml"
    script_path = f"{script_dir}/scripts_baseline/{{}}_{{}}.sh"
    model_id = "{}"
    model = "PerimidFormer"

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


    model_configs = {
        "e_layers" : [1,2,3,4,5],   # default: 1..5(original)
        "d_model" : [32,64,128,256,512,768],   # default: 32..768(original)
        "d_ff" : [512],  # only used in CLS. default: 512(original)
        "n_heads" : [8],  # default: 8(original)
        "top_k" : [2,3,4,5,6,7,8],  # default: 2..8(original)

        "moving_avg" : [0],  # default: 25(original). not used in CLS.
    }

    training_configs = {
        "is_training" : 1,
        "batch_size" : 16,  # original: 2..64
        "des" : "Exp",
        "itr" : 1,
        "dropout" : 0.1,
        "learning_rate" : 0.001,  # original: 5e-3 ~ 1e-4
        "train_epochs" : 100,  # original: 20
        "patience" : 10,  # original: 5
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
        if data_cfg["dataset"] == "Heartbeat":
            replace_dict["gpu"] = 3
            replace_dict["batch_size"] = 4  # GPU Memory Usage: 9578MiB
        if data_cfg["dataset"] == "MotorImagery":  # GPU Memory Usage: 9074MiB
            replace_dict["gpu"] = 3
        # if data_cfg["dataset"] == "DuckDuckGeese":
        #     replace_dict["batch_size"] = 8  # GPU Memory Usage: ????MiB
        #     replace_dict["gpu"] = 3
        # if data_cfg["dataset"] == "FaceDetection":
        #     replace_dict["batch_size"] = 8  # GPU Memory Usage: ????MiB
        #     replace_dict["gpu"] = 3
        # if data_cfg["dataset"] == "InsectWingbeat":
        #     replace_dict["batch_size"] = 8  # GPU Memory Usage: ????MiB
        #     replace_dict["gpu"] = 3
        # if data_cfg["dataset"] == "PEMS-SF":
        #     replace_dict["batch_size"] = 8  # GPU Memory Usage: ????MiB
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]
        
        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]
        
        
        model_cfg_tmp = model_configs.copy()
        model_configs_combination = reversed(make_combination(model_cfg_tmp))

        for model_cfg in model_configs_combination:
            if (data_key == "CLS_PenDigits") and (model_cfg["top_k"] > 5):
                continue
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