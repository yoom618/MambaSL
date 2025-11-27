import os
import math
from omegaconf import OmegaConf
import itertools

def make_combination(config_dict):
    keys, values = zip(*config_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


if __name__ == "__main__":
    script_dir = "./scripts_classification"
    data_metainfo = "data_classification_medformer.yaml"
    model = "MambaSingleLayer"

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

    exp = "additional"
    model_configs = {
        "mamba_projection_type" : ["gating"], 
        "d_model" : [32, 64, 128, 256, 512, 1024],
        "d_ff" : [1, 2, 4, 8, 16],
        "expand" : [1], 
        "d_conv" : [4], # only 2-4 available due to the implementation of causal conv1d
        "tv_dt" : [0, 1],   # 0: False, 1: True
        "tv_B" : [0, 1],    # 0: False, 1: True
        "tv_C" : [0, 1],    # 0: False, 1: True
        "use_D" : [0],    # 0: False, 1: True
    }

    training_configs = {
        "is_training" : 1,
        "itr" : 1,
        "dropout" : 0.1,
        "learning_rate" : 0.001,
        "train_epochs" : 100,
        "patience" : 10,
        "swa" : True,
    }

    os.makedirs(f"{script_dir}/scripts_mamba", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)
    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        model_configs["num_kernels"] = [max(3, math.ceil(data_cfg.seq_len / 50))]


        replace_dict = {}
        
        del data_cfg["num_class"], data_cfg["dataset"]

        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]

        model_configs_combination = reversed(make_combination(model_configs))
        
        for model_cfg in model_configs_combination:
            script_cfg = gpu_setting.copy()
            script_cfg.update(data_cfg)
            script_cfg["model"] = model
            script_cfg["model_id"] = data_key
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



        os.makedirs(f"{script_dir}/scripts_mamba/{exp}/", exist_ok=True)
        with open(f"{script_dir}/scripts_mamba/{exp}/MambaSL_{data_key}.sh", "w") as f:
            f.write(scripts)