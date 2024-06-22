from typing import Optional


def get_run_name(config: dict, name_suffix: Optional[str] = None) -> str:
    # naming the current run as <model>_<dataset>_<name_suffix>
    name_suffix = f"_{name_suffix}" if name_suffix is not None else ""
    return f"{config['model']['name']}_{config['dataset']['name']}{name_suffix}"
