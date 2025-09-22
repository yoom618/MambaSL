# InterpretGN script generation
# hyperparameters setting are referenced from below:
# 1) original InterpGN paper, code, and result csv
#    : https://github.com/YunshiWen/InterpretGatedNetwork/blob/5ea6045a1f724a536542080acaf3de3719bc4dbb/reproduce/run_uea.sh
#    : https://github.com/YunshiWen/InterpretGatedNetwork/blob/5ea6045a1f724a536542080acaf3de3719bc4dbb/result/UEA/uea_interpgn.csv


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
    model = "InterpretGN"

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
        "dnn_type" : ["FCN"],
        "num_shapelet" : [5, 10, 15], # default : [5, 10] (final=10)
        "lambda_div" : [0, 0.1, 1],  # default : [0, 0.1, 0.5, 1, 5, 10] (final=0.1)
        "lambda_reg" : [0, 0.1, 1],  # default : [0, 0.1, 0.5, 1, 5, 10] (final=0.1)
        "epsilon" : [0.5, 1, 2],  # default : [0.5, 1, 2, 5, 10] (final=1)
        "gating_value" : [0.5, 0.75, 1.0],  # default : [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] (final=1)
        # "distance_func" : ["euclidean"],
        # "memory_efficient" : [False],
        # "sbm_cls" : ["linear"],
    }

    ### while the original paper uses 5e-3 lr, beta scheduling, 500 training epochs, 50 patience, 0 dropout, 32 batch size, 
    ### we use 1e-3 lr, 100 training epochs, 10 patience, 0.1 dropout, 16 batch size for fair comparison.
    ### we know that this might harm the performance of given model in specific hyperparameter setting,
    ### but since we test the model in a wide range of hyperparameter setting, we believe that this will not affect the overall performance of the model.
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
        
        del data_cfg["num_class"], data_cfg["p_min"], data_cfg["p_max"], data_cfg["dataset"]
        
        data_cfg["root_path"] = data_cfg["root_path"].replace("data_dir", dir_setting["data_dir"])
        data_cfg["checkpoints"] = dir_setting["checkpoints"]
        
        model_configs_combination = reversed(make_combination(model_configs))
        
        for model_cfg in model_configs_combination:
            if data_cfg["dataset"] in ["Cricket", "EthanolConcentration", 
                                       "SelfRegulationSCP1", "SelfRegulationSCP2"] \
                and model_cfg["num_shapelet"] == 15:
                replace_dict["batch_size"] = 8
            if data_cfg["dataset"] in ["Heartbeat"] and model_cfg["num_shapelet"] in [10,15]:
                replace_dict["batch_size"] = 8
            if data_cfg["dataset"] in ["StandWalkJump", "PEMS-SF"]:
                if model_cfg["num_shapelet"] in [10,15]:
                    replace_dict["batch_size"] = 4
                else:
                    replace_dict["batch_size"] = 8
            # Runned in A100
            if data_cfg["dataset"] in ["MotorImagery"]:
                replace_dict["batch_size"] = 4  # 31445MiB
            if data_cfg["dataset"] in ["DuckDuckGeese"]:
                if model_cfg["num_shapelet"] in [15]:
                    replace_dict["batch_size"] = 2  # 27103MiB, 32363MiB
                else:
                    replace_dict["batch_size"] = 4  # 35951MiB
            if data_cfg["dataset"] in ["EigenWorms"]:
                if model_cfg["num_shapelet"] in [10,15]:
                    replace_dict["batch_size"] = 2  # 32363MiB
                else:
                    replace_dict["batch_size"] = 4  # 27527MiB


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