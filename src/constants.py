"""
Contains constants used throughout the project.
Don't move this file or paths will become invalid.
"""

import torch
from pathlib import Path


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Running on {DEVICE}')


#### PATHS ####
ROOT_PATH = Path(__file__).parent.parent.absolute()
DATA_PATH = ROOT_PATH.joinpath("data")
OUT_PATH = ROOT_PATH.joinpath("out")
SRC_PATH = ROOT_PATH.joinpath("src")


#### EXPERIMENTS (DO NOT CHANGE) ####
PATCH_SIZE = 16
CUTOFF = 0.25



