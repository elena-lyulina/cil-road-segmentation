from pathlib import Path

from src.constants import SRC_PATH
from src.experiments.config import generate_finetuning_config, run_config
from src.experiments.utils import get_save_path_and_experiment_name, get_run_name

# Config generated by src.experiments.config.py#generate_finetuning_config
# DON'T CHANGE THE MODELS PARAMS TO SUCCESSFULLY LOAD THE PRETRAINED MODEL
cur_config = {
    'model': {
        'name': 'small_unet',
        'params': {
            'chs': [3, 64, 128, 256, 512, 1024]
        },
        'from_pretrained': 'results/small_unet_cil_from_CIL_notebook_acc0-76_date06-07-2024_21-50-48_0.json'
    },
    'dataset': {
        'name': 'cil',
        'params': {
            'batch_size': 4,
            'shuffle': True,
            'resize_to': (384, 384)
        }
    },
    'train': {
        'n_epochs': 1,
        'optimizer': {
            'name': 'Adam',
            'params': {
                'lr': 0.0005
            }
        },
        'loss': {
            'name': 'BCELoss',
            'params': {

            }
        },
        'clip_grad': None
    }
}


if __name__ == '__main__':
    ### Uncomment to generate a fine-tuning config from a pretrained model
    # pretrained_config_path = SRC_PATH.joinpath('experiments/example/results/small_unet_cil_from_CIL_notebook_acc0-76_date06-07-2024_21-50-48_0.json')
    # generate_finetuning_config(pretrained_config_path=pretrained_config_path, dataset="cil")

    ### Run as usual
    save_path, experiment_name = get_save_path_and_experiment_name(__file__)
    run_name = get_run_name(cur_config, "finetune")
    run_config(cur_config, save_path, experiment_name, run_name, log_wandb=False, save_wandb=False)


