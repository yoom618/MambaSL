# TimesNet script generation
# hyperparameters setting are referenced from below:
# 1) time-series-library (original source code of the TimesNet paper also.)
#    : https://github.com/thuml/Time-Series-Library/blob/85c08390b6ecc5a5c3bae33b3880b8bc3e413023/scripts/classification/TimesNet.sh


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
    model = "TimesNet"

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
        # e_layers=6 is only used in the official script of PEMS-SF.
        # We found that the accuracy was lower than the paper's result with the same script. (88.24% vs 89.6%)
        # if we use e_layers=2,3,4, the highest accuracy is 87.86%.
        # The accuracy of 87.86% vs 88.24% doesn't affect the final ranking of TimesNet compared to other models,
        # and with e_layers=4 we can reduce the computational cost and also perform better in other datasets.
        # Therefore, we decided to use e_layers=2,3,4 rather than e_layers=2,3,6 in all datasets for consistency.
        "e_layers" : [2,3,4],   # default: 2(paper),  2,3,6(official script)

        ### based on the original classification script provided in tslib,
        ### it is estimated to use d_model <= d_ff.
        # "d_model" : [16,32,64],   # default: 32~64(paper),  16,32,64(official script)
        # "d_ff" : [32,64,128,256],  # default: not metioned(paper),  32,64,256(official script)
        "d_model - d_ff" : [ 
            (8,16), (8,32), (8,64),
            (16,16), (16,32), (16,64),
            (32,32), (32,64), (32,128),
            (64,64), (64,128), (64,256),
            (128,128), (128,256),
            # in some anomaly detection scripts, d_model=8 or d_model=128 is used. so we add these cases.
        ],
        "top_k" : [1,2,3],  # default: 3(paper),  1,2,3(official script)
        "num_kernels" : [4,6]  # default: not metioned(paper),  4,6(official script)
    }

    training_configs = {
        "is_training" : 1,
        "batch_size" : 16,
        "des" : "Exp",
        "itr" : 1,
        "dropout" : 0.1,
        "learning_rate" : 0.001,
        "train_epochs" : 30,
        "patience" : 10
    }
    
    os.makedirs(f"{script_dir}/scripts_baseline", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)
    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        replace_dict = {}
        ### GPU Memory Usage 
        ### AtrialFibrillation: 3062MiB / Cricket: 3644MiB / DuckDuckGeese: 6434MiB / NATOPS: 4180MiB
        if data_cfg["dataset"] == "EigenWorms":
            replace_dict["batch_size"] = 4  # GPU Memory Usage: 10246MiB
            replace_dict["gpu"] = 3
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]
        
        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]
        
        model_configs_combination = reversed(make_combination(model_configs))
        
        for model_cfg in model_configs_combination:
            script_cfg = gpu_setting.copy()
            script_cfg.update(data_cfg)
            script_cfg["model"] = model
            script_cfg["model_id"] = model_id.format(data_key)
            model_cfg["d_model"], model_cfg["d_ff"] = model_cfg["d_model - d_ff"]
            del model_cfg["d_model - d_ff"]
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