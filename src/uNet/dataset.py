import cv2
import torch
import numpy as np
from PIL import Image
from glob import glob


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


class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, image_paths, mask_paths, patch_size, cutoff, device, use_patches=True, resize_to=(400, 400)):
        self.items = []

        items = zip(image_paths, mask_paths)

        self.patch_size = patch_size
        self.cutoff = cutoff
        self.device = device
        self.resize_to = resize_to

    #     self.x, self.y, self.n_samples = None, None, None
    #
    #     self._load_data()
    #
    # def _load_data(self):  # not very scalable, but good enough for now
    #     self.x = self.images
    #     self.y = self.masks
    #     if self.use_patches:  # split each image into patches
    #         self.x, self.y = image_to_patches(self.x, self.patch_size, self.cutoff, self.y)
    #     elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
    #         self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
    #         self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
    #     self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
    #     self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        return x, y

    def __getitem__(self, item):
        # return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))
        img_path, mask_path = self.items[item]
        image = Image.open(img_path)[:3]
        mask = Image.open(mask_path).squeeze()
        if self.resize_to != (image.shape[0], image.shape[1]):  # resize images
            image = cv2.resize(image, dsize=self.resize_to)
            mask = cv2.resize(mask, dsize=self.resize_to)

        image = np.moveaxis(image, -1, 0)  # pytorch works with CHW format instead of HWC
        return self._preprocess(np_to_tensor(image, self.device), np_to_tensor(mask, self.device))

    def __len__(self):
        return len(self.items)
