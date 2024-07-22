from typing import Tuple

import torch
from sklearn.metrics import f1_score, recall_score, precision_score

from src.constants import PATCH_SIZE, CUTOFF
from src.experiments.registry import Registry

METRICS_REGISTRY = Registry()


def get_metrics() -> Tuple[dict, dict]:
    # Returns all the registered
    metrics = {'loss': [], 'val_loss': []}
    for k in METRICS_REGISTRY.keys():
        metrics[k] = []
        metrics['val_' + k] = []

    metric_fns = METRICS_REGISTRY
    return metrics, metric_fns


@METRICS_REGISTRY.register('acc')
def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


def patchify(y_hat, y):
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    #patches_hat = y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    y = torch.nn.functional.interpolate(y, (h_patches*PATCH_SIZE, w_patches*PATCH_SIZE), mode='bilinear')
    y_hat = torch.nn.functional.interpolate(y_hat, (h_patches * PATCH_SIZE, w_patches * PATCH_SIZE), mode='bilinear')
    patches_hat = torch.mean(y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE), (-1,-3)) > CUTOFF
    patches = y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF

    return patches_hat, patches


@METRICS_REGISTRY.register('patch_acc')
def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    patches_hat, patches = patchify(y_hat, y)

    return (patches == patches_hat).float().mean()


@METRICS_REGISTRY.register('patch_f1')
def path_f1_fn(y_hat, y, eps: float = 1e-10):
    patches_hat, patches = patchify(y_hat, y)

    tp = torch.sum((patches_hat == 1) & (patches == 1)).float()
    fp = torch.sum((patches_hat == 1) & (patches == 0)).float()
    fn = torch.sum((patches_hat == 0) & (patches == 1)).float()

    # Compute precision, recall, and F1 score
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)

    return f1_score


if __name__ == '__main__':
    metrics, metrics_fn = get_metrics()
    print(metrics)

