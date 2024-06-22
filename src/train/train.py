import json
import time
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import DEVICE
from src.train.metrics import get_metrics
from src.train.utils import get_optimizer, get_loss


# todo:
#  add scheduler
#  add checkpoints?

def train(
        config: dict,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        save_path: Path,
        save_name: str,
        wandb_run=None,
):
    # training loop

    # reading config
    n_epochs = config["train"]["n_epochs"]
    clip_grad = config["train"]["clip_grad"]
    optimizer = get_optimizer(config, model)
    loss_fn = get_loss(config)

    history = {}  # collects metrics at the end of each epoch

    best_val_acc = 0

    # pbar = tqdm.trange(n_epochs)
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics, metric_fns = get_metrics()

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')
        # training
        model.train()

        for (x, y) in pbar:
            optimizer.zero_grad()  # zero out gradients
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y.unsqueeze(1))
            loss.backward()  # backward pass

            if clip_grad is not None:  # clip gradients if needed
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()  # optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix({k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in val_dataloader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                y_hat = model(x)  # forward pass
                val_loss = loss_fn(y_hat, y.unsqueeze(1))

                # log partial metrics
                metrics['val_loss'].append(val_loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_' + k].append(fn(y_hat, y).item())

        # summarize and display metrics
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
            print(' '.join(['\t- ' + str(k) + ' = ' + str(v) + '\n ' for (k, v) in history[epoch].items()]))

        best_val_acc = max(history[epoch]['val_acc'], best_val_acc)

        # log to wandb
        if wandb_run:
            wandb_run.log(history[epoch])

    print('Finished Training')
    save_name = get_save_name(save_name, best_val_acc)
    save_model(config, model, optimizer, save_path, save_name, wandb_run)


def get_save_name(save_name: Optional[str], best_val_acc) -> str:
    # making sure it's unique
    return f"{save_name}_{best_val_acc}_{time.strftime('%Y%m%d-%H%M%S')}"


def save_model(config: dict, model: nn.Module, optimizer: torch.optim.Optimizer, save_path: Path, name: str, wandb_run):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = save_path.joinpath(f'{name}.json')
    model_path = save_path.joinpath(f'{name}.pth')

    # save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    # save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)

    # save to wandb
    if wandb_run:
        wandb_run.save(config_path)
        wandb_run.save(model_path)
