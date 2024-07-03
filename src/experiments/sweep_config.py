from pathlib import Path

import wandb


DEFAULT_SWEEP_CONFIG = {
    'method': 'random',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': { }
}


def get_sweep_config(
        config: dict,
        # to check other methods, go to https://docs.wandb.ai/guides/sweeps/sweep-config-keys#method
        method: str = DEFAULT_SWEEP_CONFIG['method'],
        metric_name: str = DEFAULT_SWEEP_CONFIG['metric']['name'],
        metric_goal: str = DEFAULT_SWEEP_CONFIG['metric']['goal']
) -> dict:
    sweep_config = DEFAULT_SWEEP_CONFIG
    sweep_config['method'] = method
    sweep_config['metric']['name'] = metric_name
    sweep_config['metric']['goal'] = metric_goal
    sweep_config['parameters'] = config
    return sweep_config


def run_sweep_config(
        sweep_config: dict,
        save_path: Path,
        experiment_name: str,
        run_name: str,
        sweep_id: str = None,
):

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep_config, project=SWEEP_PROJECT)




def init_sweep(sweep_project: str, sweep_id: str, config: dict):
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project=SWEEP_PROJECT)


# run $ python3 sweep_mamba.py 1
if __name__ == "__main__":
    # init_sweep()
    parser = ArgumentParser()
    parser.add_argument('count', type=int)
    args = parser.parse_args()
    wandb.agent('x0xbmmrl', train, count=args.count, project=SWEEP_PROJECT)

