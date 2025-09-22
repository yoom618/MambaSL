# TimeMixer++ script generation
# hyperparameters setting are referenced from below:
# 1) since TimeMixer++ isn't open-sourced yet (@2025/04/13),
#    we refer to the paper and their anonymous repo (https://anonymous.4open.science/r/TimeMixerPP/, accessed on 2025/04/13)
#    however, the repo only provides forecasting task scripts


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
    model = "TimeMixerPP"

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
        # Used for making embedding
        "down_sampling_method" : ['conv'],  # default: 'conv'
        "down_sampling_layers" : [1, 2, 3],  # default: 2, 3. 
        #                                      The original paper mentioned that 1 to 3 is enough for overall performance and efficiency
        "down_sampling_window" : [2],  # default: 2
        
        # MultiScaleTrendCross & MultiScaleSeasonalCross
        "num_kernels" : [6],  # default: 6
        
        # MixerBlock
        "e_layers" : [1, 2, 3, 4],  # default: 2. 
        #                             The original paper tested 1 to 4 and mentioned that more layers typically lead to better performance.
        #                             However, since classification datasets all have different complexity, we set it from 1 to 4.
        "d_model" : [16, 32, 64],  # default: 16, 32
        "d_ff" : [16, 32, 64],  # default: 32, 64
        "top_k" : [2, 3],  # default: 3.

        # # Below flags are not used in classification task
        # "channel_mixing" : [0],
        # "channel_independence" : [0],
        # "output_attention" : [False],
        # "n_heads" : [4],
    }

    training_configs = {
        "is_training" : 1,
        "batch_size" : 16,
        "des" : "Exp",
        "itr" : 1,
        "dropout" : 0.1,
        "learning_rate" : 0.001,
        "train_epochs" : 100,  # defaults are less than 10 in forecast task, which is similar to other transformer-based models. thus set to 100 for fair comparison
        "patience" : 10,
    }
    
    os.makedirs(f"{script_dir}/scripts_baseline", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)

    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        replace_dict = {}
        # AtrialFibrillation: 4120MiB / Cricket: 2636MiB / EthanolConcentration: 3708MiB / SCP2: 6532MiB / MotorImagery: 6384MiB
        if data_cfg["dataset"] == "EigenWorms":
            replace_dict["batch_size"] = 4  # >33939MiB, cannot run on 1080ti with batch_size=1 & cannot run on A100 with batch_size=16
            replace_dict["gpu"] = 0
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]
        
        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]
        
        model_configs_combination = reversed(make_combination(model_configs))
        
        for model_cfg in model_configs_combination:
            if data_cfg["dataset"]=="PenDigits" and model_cfg["down_sampling_layers"]==3:
                # PenDigits: cannot run with 3 downsampling layers since its seq_len is 8
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