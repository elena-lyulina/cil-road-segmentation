import sys
sys.path.insert(1, "/home/guptav/cil-road-segmentation/")
from src.experiments.config import run_config
from src.experiments.utils import get_run_name, get_save_path_and_experiment_name

cur_config = {
    'model': {
        'name': 'Unet',
        'params': {
            'encoder_name': 'resnet101',
            'padding_mode': 'edge',
            'classes': 2,
            'activation': 'sigmoid',
            'aux_params': None
        },
    },
    'dataset': {
        'name': 'cil',
        'params': {
            'batch_size': 4,
            'num_workers': 4,
            'shuffle': True,
            'resize_to': (400, 400),
            'augment': ['geometric']
        }
    },
    'train': {
        'n_epochs': 10,
        'optimizer': {
            'name': 'Adam',
            'params': {
                'lr': 0.0002
            }
        },
        'loss': 'DiceBCELoss',
        'clip_grad': 1,
        'n_gpus': 1
    }
}

if __name__ == '__main__':
    save_path, experiment_name = get_save_path_and_experiment_name(__file__)
    run_name = get_run_name(cur_config, "baseline")

    run_config(cur_config, save_path, experiment_name, run_name, log_wandb=True)
