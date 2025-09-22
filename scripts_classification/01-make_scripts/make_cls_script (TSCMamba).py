# TSCMamba script generation
# hyperparameters setting are referenced from below:
# 1) original TSCMamba script
#    : https://github.com/Atik-Ahamed/TSCMamba/blob/816ab099c2e4fd0fad5b8b206963c8f80dd7a457/scripts/classification/TSCMamba.sh

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
    model = "TSCMamba"

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
    # gpu_setting = {
    #     "use_multi_gpu" : True,
    #     "devices" : "1,2"
    # }


    model_configs = {
        "d_model" : [64,128,256],  # default: 64 (named `projected_space` in the original code) (tested in ablation study: 64~1024)
        "e_layers" : [1,2],  # default: 1 (named `num_mambas` in the original code)
        "expand" : [1,2],  # default: 2 (named `e_fact` in the original code) (tested in ablation study: 1~5)
        "d_conv" : [2,4],  # default: 4 (tested in ablation study: 2,4. actually it can be either 2,3,4 due to causal_conv1d implementation)
        "d_ff" : [32,64,128],  # default: 128 (=d_state) (tested in ablation study: 16~256)
        "half_rocket" : [0,1],  # default: 0. this is only used when `no_rocket` is 0
        "additive_fusion" : [0,1],  # default: 1. (1: additive fusion, 0: multiplicative fusion)
            

        ### Fixed as the paper's eqational expressions
        "only_forward_scan" : [0],  # default: 0 (should be set to 0 to use Tango Scanning)
        "flip_dir" : [2], # default: 2 (1: horizontal flip, 2: vertical flip. this should be set to 0 as the paper's eq(8))
        "reverse_flip" : [0],  # default: 0. (this should be set to 0 as the paper's eq(10))
        
        ### Fixed as the paper's hyperparameters
        "patch_size" : [8],  # default: 8 (They mentioned that it was fixed to 8 in the paper)
                             # quote) "where p = 8 is patch size"
        "rescale_size" : [64],  # default: 64 (`L1` variable in the original paper. They mentioned it was fixed to 64 in the paper)
                                # idk whether this setting is okay with datasets with sequence length less than 64.
                                # quote) "In this paper, we adopt L1 = 64 for computational efficiency and expressiveness of the obtained wavelet features."
        "variation" : [64],  # default: 64 (set to the same value as rescale_size, since it doesn't have to be either maller or larger than L1)
        "initial_focus" : [1.0],  # default: 1.0 (initial value of learnable focus parameter. They mentioned it was fixed to 1.0 in the paper)
                                  # quote) "The initial value of λ= 1.0 ensures a balanced focus initially between temporal and spectral domain features."


        ### Parameters that are provided from the original TSCMamba script is quite different from the paper's final structure n results.
        ### We've already asked the first author about these parameters, 
        ### and he confirmed that our assumptions are correct, and the provided script is merely an example for running the model.
        ### Therefore, we set these parameters based on our understanding of the paper's model design.
        # 1. `no_rocket` : The paper's model is basically designed to use ROCKET (as can be seen in the figure 1)
        #                  but the original script sets `no_rocket` to 1 (which means no ROCKET).
        #                  However since the script is just an example for running the model,
        #                  Since we think using ROCKET is one of the main contributions of the paper,
        #                  we set `no_rocket` to 0, which means the model will use ROCKET features.
        #                  Instead, we set `half_rocket` to either 0 or 1 to see the effect of using original feature.
        "no_rocket" : [0], # default: 1

        # 2. `max_pooling`` : The paper's ablation study shows that avg_pooling is better than max_pooling.
        #                     However, the original script sets this parameter to 1 not 0.
        #                     The author replied that using either avg_pooling or max_pooling is their contribution,
        #                     and they just set max_pooling to 1 in the original script.
        #                     So we set `max_pooling` to either 0 or 1 to get both results.
        "max_pooling" : [0,1], # default: 1


        ### Unknown
        "channel_token_mixing" : [0],  # default: 0 (This parameter is not mentioned in the paper)
                                       # When it is on, rocket features are generated half from channel axis and half from token axis.
                                       # Since channel n token mixing is implemented inside TSCMamba (using two mamba blocks), we set this to 0 for unintended mixing behavior.
    }

    training_configs = {
        "is_training" : 1,
        "batch_size" : 16,  # default: 32 but set to 16 for fair comparison
        "des" : "Exp",
        "itr" : 1,
        "dropout" : 0.1,  # default: 0.2 but set to 0.1 for fair comparison
        "learning_rate" : 0.001,  # default: 0.0001 but set to 0.001 for fair comparison
        # "weight_decay" : 0.0,  # default: use cosine decay but we set to 0.0 (default) for fair comparison
        "train_epochs" : 100,  # default: 200 but set to 100 for fair comparison
        "patience" : 10,  # default: 100 but set to 10 for fair comparison
    }

    os.makedirs(f"{script_dir}/scripts_baseline", exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)
    
    for data_key, data_cfg in data_configs.items():
        scripts = ""

        # Fix kernel size to max(3, 1%*seq_len) in Embedding step
        # Use default d_conv size in MambaBlock
        model_configs["num_kernels"] = [max(3, math.ceil(data_cfg.seq_len / 50))]
        model_configs["d_conv"] = [4]

        # Set d_state(=d_ff) from log2(1/p_min) - 1  to  log2(1/p_min) + 1
        model_configs["d_ff"] = [max(2, math.ceil(math.log2(1/data_cfg.p_min))) + i for i in range(-1, 2)]
        

        replace_dict = {
            "data" : "UEA4TSCMamba",
        }
        # PEMS-SF: 5304MiB
        
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

            script = f"python run.py \\\n"
            for key, value in script_cfg.items():
                script += f"  --{key} {value} \\\n"
            script = script[:-3]
            script += "\n\n"
            scripts += script
            print(script)

        with open(script_path.format(model, data_key), "w") as f:
            f.write(scripts)