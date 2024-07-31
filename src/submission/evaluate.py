from glob import glob
from pathlib import Path

from tqdm import tqdm
from wandb.wandb_torch import torch

from src.constants import DEVICE, ROOT_PATH
from src.data.datahandler import get_datahandler_class
from src.experiments.config import get_model_path_from_config, load_config
from src.train.metrics import get_metrics
from src.train.train import load_checkpoint


def get_all_configs(folders: list, name_keyword='', exclude_keywords: list = None, extension='json'):
    all_configs = []
    for folder in folders:
        all_configs += glob(folder + f'/**/*{name_keyword}*.{extension}', recursive=True)

    if exclude_keywords is not None:
        all_configs = [c for c in all_configs if all(k not in c for k in exclude_keywords)]
    return sorted(all_configs)


def evaluate(config_paths: list, dataset_name: str, dataset_params=None):
    for config_path in config_paths:
        config = load_config(Path(config_path))
        model_path = get_model_path_from_config(Path(config_path))
        model_name = config['model']['name']
        if model_name == "end2end":
            print('Not sure we can evaluate end2end here')
            continue

        print(f'Evaluating model {model_name} from config {Path(config_path).name} on dataset {dataset_name}')
        model, _ = load_checkpoint(model_path)
        model.eval()

        dataset_params = {} if dataset_name is not None else dataset_params

        datahander = get_datahandler_class(dataset_name)(**dataset_params)
        _, val_dataloader = datahander.get_train_val_dataloaders()

        # initialize metric list
        metrics, metric_fns = get_metrics()

        # validation
        model.eval()
        val_pbar = tqdm(val_dataloader, desc=f'Validation')

        for i, batch in enumerate(val_pbar):
            if len(batch) == 2:
                x, y = batch
                cluster_id = None  # or some default value if needed
            elif len(batch) == 3:
                x, y, cluster_id = batch
            else:
                raise ValueError("Unexpected batch size: expected 2 or 3 items, got {}".format(len(batch)))

            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            if model_name == "end2end":
                y_hat = model((x, cluster_id))
            else:
                y_hat = model(x)  # forward pass

            # log partial metrics
            for k, fn in metric_fns.items():
                metrics['val_' + k].append(fn(y_hat, y).item())
        metrics = {k: v for k, v in metrics.items() if len(v) > 0}
        # summarize and display metrics
        metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        print(' '.join(['\t- ' + str(k) + ' = ' + str(v) + '\n ' for (k, v) in metrics.items()]))
        print('\n')


if __name__ == '__main__':
    path = str(ROOT_PATH.joinpath('out', 'models'))

    configs = get_all_configs([path], name_keyword='cluster0')
    evaluate(configs, 'cil_cluster0')
    print('\n')

    configs = get_all_configs([path], name_keyword='cluster1')
    evaluate(configs, 'cil_cluster1')
    print('\n')

    configs = get_all_configs([path], exclude_keywords=['cluster0', 'cluster1'])
    evaluate(configs, 'cil')



