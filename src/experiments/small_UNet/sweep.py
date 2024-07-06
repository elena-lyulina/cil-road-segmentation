import json

from src.experiments.sweep_config import get_sweep_config, init_sweep, run_sweep_agent
from src.experiments.utils import get_save_path_and_experiment_name

### Step 1. modify the parameteres to tune
cur_config = {
    'model': {
        'name': 'small_unet',
        'params': {
            'chs': (3, 64, 128, 256, 512, 1024)
        }
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
                "SWEEP_lr": {  # a distribution of possible values for sweep
                    'distribution': 'uniform',
                    'min': 0.001,
                    'max': 0.1
                }
            }
        },
        'loss': {
            'name': 'BCELoss',
            'params': { }
        },
        'SWEEP_clip_grad': { # a list of possible values for sweep
            'values': [None, 1]
        }
    }
}


if __name__ == '__main__':
    save_path, experiment_name = get_save_path_and_experiment_name(__file__)

    ### Step 2, uncomment to init the sweep
    # sweep_config = get_sweep_config(cur_config)
    # print(json.dumps(sweep_config, indent=4))
    # init_sweep(sweep_config, experiment_name)

    ### Step 3, insert the sweep_id from the output of the previous step, e.g.
    # sweep_id = '3vv027ep'

    ### Step 4, uncomment to make a few runs for the current sweep
    # run_sweep_agent(cur_config, sweep_id, 1, experiment_name, save_path)



