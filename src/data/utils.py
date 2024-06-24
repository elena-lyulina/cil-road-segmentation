from glob import glob

import numpy as np
import torch
from PIL import Image

from src.experiments.registry import Registry

DATASET_REGISTRY = Registry()


def get_dataset_class(dataset_name: str):
    # Returns dataset's class registered under the given name
    cls = DATASET_REGISTRY.get(dataset_name)
    if cls is None:
        raise ValueError(f'Dataset {dataset_name} is not registered')
    return cls


### The following code is taken from the CIL notebook ###

def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.stack([np.array(Image.open(f)) for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.


def image_to_patches(images, patch_size, cutoff, masks=None):
    # takes in a 4D np.array containing images and (optionally) a 4D np.array containing the segmentation masks
    # returns a 4D np.array with an ordered sequence of patches extracted from the image and (optionally) a np.array containing labels
    n_images = images.shape[0]  # number of images
    h, w = images.shape[1:3]  # shape of images
    assert (h % patch_size) + (w % patch_size) == 0  # make sure images can be patched exactly

    images = images[:, :, :, :3]

    h_patches = h // patch_size
    w_patches = w // patch_size

    patches = images.reshape((n_images, h_patches, patch_size, w_patches, patch_size, -1))
    patches = np.moveaxis(patches, 2, 3)
    patches = patches.reshape(-1, patch_size, patch_size, 3)
    if masks is None:
        return patches

    masks = masks.reshape((n_images, h_patches, patch_size, w_patches, patch_size, -1))
    masks = np.moveaxis(masks, 2, 3)
    labels = np.mean(masks, (-1, -2, -3)) > cutoff  # compute labels
    labels = labels.reshape(-1).astype(np.float32)
    return patches, labels


def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)
