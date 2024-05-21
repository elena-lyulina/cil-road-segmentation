import os

import torch
from sklearn.model_selection import train_test_split
from torch import nn

from dataset import load_all_from_path, ImageDataset
from src.uNet.model import UNet
from model import train


def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()

def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    patches_hat = y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    patches = y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    return (patches == patches_hat).float().mean()


if __name__ == '__main__':
    PATCH_SIZE = 16
    VAL_SIZE = 10
    CUTOFF = 0.25

    ROOT_PATH = "../../"
    images = load_all_from_path(os.path.join(ROOT_PATH, 'data', 'training', 'images'))[:, :, :, :3]
    masks = load_all_from_path(os.path.join(ROOT_PATH, 'data', 'training', 'groundtruth'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    train_dataset = ImageDataset(train_images, train_masks, PATCH_SIZE, CUTOFF, device, use_patches=False, resize_to=(384, 384))
    val_dataset = ImageDataset(val_images, val_masks, PATCH_SIZE, CUTOFF, device, use_patches=False, resize_to=(384, 384))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)
    model = UNet().to(device)
    loss_fn = nn.BCELoss()
    metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 5
    train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)