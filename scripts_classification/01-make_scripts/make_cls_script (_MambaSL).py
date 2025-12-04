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
    model = "MambaSingleLayer"

    dir_setting = {
        "data_dir" : "/data/user/MambaSL/dataset",
        "checkpoints": "/data/user/MambaSL/checkpoints",
    }

    data_configs = OmegaConf.load(f"{script_dir}/{data_metainfo}")

    gpu_setting = {
        "use_gpu" : True,
        "gpu_type" : "cuda",
        "gpu" : 2
    }

    exp = "proposed"
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
    # exp = "proposedTB"
    # model_configs = {
    #     "mamba_projection_type" : ["gating"], 
    #     "d_model" : [32, 64, 128, 256, 512, 1024],
    #     "d_ff" : [1, 2, 4, 8, 16],
    #     "expand" : [1], 
    #     "d_conv" : [4], # only 2-4 available due to the implementation of causal conv1d
    #     "tv_dt" : [0, 1],   # 0: False, 1: True
    #     "tv_B" : [0, 1],    # 0: False, 1: True
    #     "tv_C" : [0, 1],    # 0: False, 1: True
    #     "use_D" : [0],    # 0: False, 1: True
    #     "save_log" : [True]
    # }



    # # ablation1 : original Mamba model
    # exp = "ablation1"
    # model = "Mamba"
    # model_configs = {
    #     "mamba_projection_type" : ["full"], 
    #     "d_model" : [32, 64, 128, 256, 512, 1024],
    #     "d_ff" : [1, 2, 4, 8, 16],
    #     "expand" : [1], 
    #     "d_conv" : [4],
    # }


    # # ablation2 : change embedding kernel size (2%seq_len -> 3)
    # exp = "ablation2"
    # model_configs = {
    #     "mamba_projection_type" : ["gating"], 
    #     "num_kernels" : [3],  # different from proposed model
    #     "d_model" : [32, 64, 128, 256, 512, 1024],
    #     "d_ff" : [1, 2, 4, 8, 16],
    #     "expand" : [1], 
    #     "d_conv" : [4],
    #     "tv_dt" : [0, 1],
    #     "tv_B" : [0, 1],
    #     "tv_C" : [0, 1],
    #     "use_D" : [0],
    # }


    # # ablation3 : change skip connection (False -> True)
    # exp = "ablation3"
    # model_configs = {
    #     "mamba_projection_type" : ["gating"], 
    #     "d_model" : [32, 64, 128, 256, 512, 1024],
    #     "d_ff" : [1, 2, 4, 8, 16],
    #     "expand" : [1], 
    #     "d_conv" : [4],
    #     "tv_dt" : [0, 1],
    #     "tv_B" : [0, 1],
    #     "tv_C" : [0, 1],
    #     "use_D" : [1],  # different from proposed model
    # }
    # # ablation3TB : change skip connection (False -> True) + TensorBoard logging
    # exp = "ablation3TB"
    # model_configs = {
    #     "mamba_projection_type" : ["gating"], 
    #     "d_model" : [32, 64, 128, 256, 512, 1024],
    #     "d_ff" : [1, 2, 4, 8, 16],
    #     "expand" : [1], 
    #     "d_conv" : [4],
    #     "tv_dt" : [0, 1],
    #     "tv_B" : [0, 1],
    #     "tv_C" : [0, 1],
    #     "use_D" : [1],  # different from proposed model
    #     "save_log" : [True]
    # }

    # # ablation4 : change adjusting time-varying features (fix tvs to 1)
    # exp = "ablation4"
    # model_configs = {
    #     "mamba_projection_type" : ["gating"], 
    #     "d_model" : [32, 64, 128, 256, 512, 1024],
    #     "d_ff" : [1, 2, 4, 8, 16],
    #     "expand" : [1, 2, 4],  # different from proposed model
    #     "d_conv" : [1, 2, 4],  # different from proposed model
    #     "tv_dt" : [1],  # different from proposed model
    #     "tv_B" : [1],  # different from proposed model
    #     "tv_C" : [1],  # different from proposed model
    #     "use_D" : [0],
    # }

    # # ablation5 : change aggregation type (full, max, avg, last)
    # exp = "ablation5"
    # model_configs = {
    #     "mamba_projection_type" : ["full", "max", "avg", "last"],
    #     "d_model" : [32, 64, 128, 256, 512, 1024],
    #     "d_ff" : [1, 2, 4, 8, 16],
    #     "expand" : [1], 
    #     "d_conv" : [4],
    #     "tv_dt" : [0, 1],
    #     "tv_B" : [0, 1],
    #     "tv_C" : [0, 1],
    #     "use_D" : [0],
    # }


    # # additinal : only use tv features (additional testing since others already showed good performance)
    # exp = "additional"
    # model_configs = {
    #     "mamba_projection_type" : ["avg"],  # different from proposed model
    #     "num_kernels" : [3],  # different from proposed model
    #     "d_model" : [32, 64, 128, 256, 512, 1024],
    #     "d_ff" : [1, 2, 4, 8, 16],
    #     "expand" : [1], 
    #     "d_conv" : [4],
    #     "tv_dt" : [0, 1],
    #     "tv_B" : [0, 1],
    #     "tv_C" : [0, 1],
    #     "use_D" : [1],  # different from proposed model
    # }



    # additinal : use train dataset as validation dataset (similar to InceptionTime setting)
    exp = "trainlossonly"
    model_configs = {
        "mamba_projection_type" : ["gating"], 
        "d_model" : [32, 64, 128, 256, 512, 1024],
        "d_ff" : [1, 2, 4, 8, 16],
        "expand" : [1], 
        "d_conv" : [4],
        "tv_dt" : [0, 1],
        "tv_B" : [0, 1],
        "tv_C" : [0, 1],
        "use_D" : [0],
    }

    if exp == "trainlossonly":
        training_configs = {
            "task_name" : "classification_trainlossonly",
            "is_training" : 1,
            "batch_size" : 16,
            "des" : f"{exp}",
            "itr" : 1,
            "dropout" : 0.1,
            "learning_rate" : 0.001,
            "train_epochs" : 100,
            "patience" : 10,
        }
    else:
        training_configs = {
            "is_training" : 1,
            "batch_size" : 16,
            "des" : f"{{}}4{exp}",
            "itr" : 1,
            "dropout" : 0.1,
            "learning_rate" : 0.001,
            "train_epochs" : 100,
            "patience" : 10,
        }

    os.makedirs(f"{script_dir}/scripts_mamba", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)
    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        if exp not in ["ablation1", "ablation2"]:
            # Fix kernel size to max(3, 2%*seq_len) in Embedding step
            model_configs["num_kernels"] = [max(3, math.ceil(data_cfg.seq_len / 50))]


        replace_dict = {}
        if data_cfg["dataset"][0].lower() < "f":
            replace_dict["gpu"] = 0
        elif data_cfg["dataset"][0].lower() < "n":
            replace_dict["gpu"] = 1
        else:
            replace_dict["gpu"] = 2
        if data_cfg["dataset"] == "EigenWorms":
            replace_dict["batch_size"] = 4  # GPU Memory Usage: 9136MiB
            replace_dict["gpu"] = 3
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]

        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]

        model_configs_combination = reversed(make_combination(model_configs))
        
        for model_cfg in model_configs_combination:
            script_cfg = gpu_setting.copy()
            script_cfg.update(data_cfg)
            script_cfg["model"] = model
            script_cfg["model_id"] = data_key
            script_cfg.update(model_cfg)
            replace_dict["des"] = training_configs["des"].format(model_cfg["mamba_projection_type"])
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
        with open(f"{script_dir}/scripts_mamba/{exp}/MambaSL_{data_key}_{exp}.sh", "w") as f:
            f.write(scripts)