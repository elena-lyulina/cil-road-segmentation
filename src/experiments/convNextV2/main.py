from src.experiments.config import run_config
from src.experiments.utils import get_run_name, get_save_path_and_experiment_name

cur_config = {
    'model': {
        'name': 'convNextV2',
        'params': {
            'num_classes': 2
        }
    },
    'dataset': {
        'name': 'cil',
        'params': {
            'batch_size': 4,
            'num_workers': 4,
            'shuffle': True,
            'resize_to': (400, 400),
            'augment': 'masked',
            'masking_params': {
                'num_zero_patches': 8,
                'zero_patch_size': 50,
                'num_flip_patches': 25,
                'flip_patch_size': 16
            }
        }
    },
    'train': {
        'n_epochs': 5,
        'optimizer': {
            'name': 'Adam',
            'params': {
                'lr': 0.0005
            }
        },
        'loss': {
            'name': 'ftBCE',
            'params': {

            }
        },
        'clip_grad': None,
        'n_gpus': 1
    }
}

if __name__ == '__main__':
    save_path, experiment_name = get_save_path_and_experiment_name(__file__)
    run_name = get_run_name(cur_config, "from_CIL_notebook")

    run_config(cur_config, save_path, experiment_name, run_name, log_wandb=False)
