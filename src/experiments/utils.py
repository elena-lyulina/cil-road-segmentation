from pathlib import Path
from typing import Optional, Tuple


def get_run_name(config: dict, name_suffix: Optional[str] = None) -> str:
    # naming the current run as <model>_<dataset>_<name_suffix>
    name_suffix = f"_{name_suffix}" if name_suffix is not None else ""
    return f"{config['model']['name']}_{config['dataset']['name']}{name_suffix}"


def get_save_path_and_experiment_name(file) -> Tuple[Path, str]:
    # saving trained models to "results" folder in this experiment's folder
    save_path = Path(file).parent.absolute().joinpath('results')
    # naming the experiment by this experiment's folder
    experiment_name = Path(file).parent.name

    return save_path, experiment_name
