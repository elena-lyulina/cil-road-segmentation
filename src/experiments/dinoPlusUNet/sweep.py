import json
cur_config = {
    'model': {
        'name': 'dino_plus_unet',
        'params': {}
    },
    'dataset': {
        'name': 'all',
        'params': {
            'batch_size': 16,
            'num_workers': 4,
            'shuffle': True,
            'resize_to': (400, 400),
            'augment': ['geometric']
        }
    },
    'train': {
        'n_epochs': 2,
        'optimizer': {
            'name': 'Adam',
            'params': {
                "SWEEP_lr": {
                        'distribution': 'uniform',
                        'min': 0.0001,
                        'max': 0.01
                }
            }
        },
        'SWEEP_loss': {'values': [
            'BCELoss',
            'sDice',
            'lcDice',
            'sqDice',
            'ft',
            'DiceBCELoss',
            'lcDiceBCELoss',
            'sqDiceBCELoss',
            'ftBCE'
        ]},

        'SWEEP_clip_grad': { # a list of possible values for a sweep
            'values': [None, 1]
        },
        'n_gpus': 1
    }
}

from src.experiments.sweep_config import get_sweep_config, init_sweep, run_sweep_agent
from src.experiments.utils import get_save_path_and_experiment_name

### Step 1. generate a usual config and modify the parameteres to tune



if __name__ == '__main__':
    save_path, experiment_name = get_save_path_and_experiment_name(__file__)

    ## Step 2, uncomment to init the sweep
    sweep_config = get_sweep_config(cur_config)
    print(json.dumps(sweep_config, indent=4))
    # init_sweep(sweep_config, experiment_name)

    ## Step 3, insert the sweep_id from the output of the previous step, e.g.
    sweep_id = ''

    ## Step 4, uncomment to make a few runs for the current sweep
    # run_sweep_agent(cur_config, sweep_id, 10, experiment_name, save_path)



