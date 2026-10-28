import itertools
import math
import os

from omegaconf import OmegaConf


def make_combination(config_dict):
    keys, values = zip(*config_dict.items())
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


if __name__ == "__main__":
    script_dir = "./scripts_classification"
    data_configs = OmegaConf.load(f"{script_dir}/data_classification.yaml")

    dir_setting = {
        "data_dir": "/data/yoom618/TSLib/dataset",
        "checkpoints": "/data/yoom618/TSLib/checkpoints",
    }
    output_dir = f"{script_dir}/scripts_mamba/Mamba3SL_fullgrid"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dir_setting["checkpoints"], exist_ok=True)

    # Keep only values used by the final MambaSL scripts. Mamba-3 requires
    # d_state >= 4, so the MambaSL d_ff values 1 and 2 are not valid here.
    model_configs = {
        "mamba_projection_type": ["gating"],
        "d_model": [32, 64, 128, 256, 512, 1024],
        "d_ff": [4, 8, 16],
        "expand": [1],
        "d_conv": [4],
        "tv_dt": [0, 1],
        "tv_B": [0, 1],
        "tv_C": [0, 1],
        "tv_A": [0, 1],  # added for Mamba3
        "use_D": [0],
    }
    combinations = make_combination(model_configs)
    combinations.reverse()
    training_configs = {
        "task_name": "classification",
        "is_training": 1,
        "batch_size": 16,
        "des": "gating4proposed",
        "itr": 1,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "train_epochs": 100,
        "patience": 10,
    }
    gpu_setting = {
        "use_gpu": True,
        "gpu_type": "cuda",
    }

    for data_key, data_cfg in data_configs.items():
        dataset = data_cfg["dataset"]
        dataset_cfg = dict(data_cfg)
        for key in ("num_class", "p_min", "p_max", "dataset"):
            dataset_cfg.pop(key)

        dataset_cfg["root_path"] = dataset_cfg["root_path"].replace(
            "data_dir", dir_setting["data_dir"]
        )
        dataset_cfg["checkpoints"] = dir_setting["checkpoints"]
        kernel_size = max(3, math.ceil(data_cfg["seq_len"] / 50))

        # if dataset[0].lower() < "f":
        #     gpu_id = 0
        # elif dataset[0].lower() < "n":
        #     gpu_id = 1
        # else:
        #     gpu_id = 2
        # batch_size = 16
        # if dataset == "EigenWorms":
        #     gpu_id = 3
        #     batch_size = 4

        gpu_id = 0
        batch_size = 16
        if dataset == "EigenWorms":
            gpu_id = 3
            batch_size = 4

        script_lines = []
        for model_cfg in combinations:
            script_cfg = {
                **gpu_setting,
                **dataset_cfg,
                "model": "Mamba3SL",
                "model_id": data_key,
                **model_cfg,
                "num_kernels": kernel_size,
                **training_configs,
                "batch_size": batch_size,
                "gpu": gpu_id,
            }
            script_lines.append("python run.py \\")
            for key, value in script_cfg.items():
                script_lines.append(f"  --{key} {value} \\")
            script_lines[-1] = script_lines[-1][:-2]
            script_lines.append("")

        output_path = os.path.join(
            output_dir, f"Mamba3SL_{data_key}_proposed.sh"
        )
        with open(output_path, "w", encoding="utf-8") as script_file:
            script_file.write("\n".join(script_lines))

        print(
            f"Wrote {len(combinations)} configurations for {dataset} "
            f"to {output_path}"
        )
