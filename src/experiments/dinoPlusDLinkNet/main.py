from pathlib import Path
import sys

sys.path.insert(0, "/home/guptav/cil-road-segmentation/")
from src.experiments.config import run_config
from src.experiments.utils import get_run_name, get_save_path_and_experiment_name

# Config generated by src.experiments.config.py#generate_config
cur_config = {
    "model": {
        "name": "dino_plus_dlinknet",
        "params": {},
    },
    "dataset": {
        "name": "cil",
        "params": {"batch_size": 8, "num_workers": 4, "shuffle": True, "resize_to": (384, 384), "augment": ["geometric"]}
    },
    "train": {
        "n_epochs": 5,
        "optimizer": {"name": "Adam", "params": {"lr": 0.01}},
        "loss": "BCELoss",
        "clip_grad": None,
        "n_gpus": 1
    },
}


if __name__ == "__main__":
    save_path, experiment_name = get_save_path_and_experiment_name(__file__)
    run_name = get_run_name(cur_config, "test")

    run_config(cur_config, save_path, experiment_name, run_name, log_wandb=False)