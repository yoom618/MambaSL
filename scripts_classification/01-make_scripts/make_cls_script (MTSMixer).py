# MTS-Mixer script generation
# hyperparameters setting are referenced from below:
# 1) Original MTS-Mixer source code (only for forecasting task)
#    : https://github.com/plumprc/MTS-Mixers/blob/f20ee27b8b6dba213f7d6a4dc55de3f44c2d181e/script.md

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
    model = "MTSMixer"

    dir_setting = {
        "data_dir" : "/data/yoom618/TSLib/dataset",
        "checkpoints": "/data/yoom618/TSLib/checkpoints",
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
        "d_model" : [128, 256, 512, 1024],  # default: 256, 512, 1024
        "use_norm" : [1],  # default: 1(True)
        "e_layers" : [2], # default: 2

        # d_ff is for FactorizedChannelMixing & d_ff should be smaller than enc_in
        # where enc_in is between 2 ~ 1345 in UEA datasets
        # 0 means fac_C is False, else fac_C is True
        "d_ff" : [0, 2, 4, 8, 16, 32, 64, 128],  # default: 16, 64. available only when fac_C is True

        # down_sampling_window is for FactorizedTemporalMixing & should be smaller than seq_len
        # since it was set from [1, 2, 3, 4, 6, 8, 12] in the original paper's forecast task (seq_len = 36 or 96),
        # we set it to the similar range considering the seq_len of UEA datasets (seq_len = 8 ~ 17984).
        # ** it doesn't have to be the divisors of seq_len since we modified the model a bit **
        # 0 means fac_T is False, else fac_T is True
        "down_sampling_window_ratio": [0, 1, 2, 3, 5, 7.5, 10, 12.5,],  # default: 2, 3, 6, 8. available only when fac_T is True
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
        # if data_cfg["dataset"] == "EigenWorms":  # GPU Memory Usage: 4852MiB
        #     replace_dict["batch_size"] = 16
        if data_cfg["dataset"] == "DuckDuckGeese":
            replace_dict["batch_size"] = 8  # GPU Memory Usage: 6420MiB
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]
        
        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]
        
        model_cfg_tmp = model_configs.copy()
        model_cfg_tmp["d_ff"] = list(filter(lambda x: x < data_cfg["enc_in"], model_configs["d_ff"]))
        model_cfg_tmp["down_sampling_window"] = sorted(set([math.ceil(data_cfg["seq_len"] * (ratio/100)) for ratio in model_configs["down_sampling_window_ratio"]]))
        model_cfg_tmp.pop("down_sampling_window_ratio")
        model_configs_combination = reversed(make_combination(model_cfg_tmp))
        
        for model_cfg in model_configs_combination:
            if model_cfg["d_ff"] == 0:
                model_cfg["fac_C"] = False
            else:    
                model_cfg["fac_C"] = True
            
            if model_cfg["down_sampling_window"] == 0:
                model_cfg["fac_T"] = False
            else:
                model_cfg["fac_T"] = True

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