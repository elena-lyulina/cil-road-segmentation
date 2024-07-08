from glob import glob
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
import torch

from src.constants import DATA_PATH, DEVICE, CUTOFF, PATCH_SIZE
from src.data.datahandler import DATAHANDLER_REGISTRY, DataHandler
from src.data.utils import DATASET_REGISTRY, np_to_tensor


@DATAHANDLER_REGISTRY.register("cil")
class CILDataHandler(DataHandler):
    dataset_path = DATA_PATH.joinpath("cil")

    train_path = dataset_path.joinpath("training")
    train_images_path = train_path.joinpath("images")
    train_masks_path = train_path.joinpath("groundtruth")

    def __init__(self, batch_size=4, shuffle=True, resize_to=(400, 400), augment=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize_to = resize_to
        self.augment = augment

        images_paths = [f for f in sorted(glob(str(self.train_images_path) + "/*.png"))]
        masks_paths = [f for f in sorted(glob(str(self.train_masks_path) + "/*.png"))]

        (
            self.train_image_paths,
            self.val_image_paths,
            self.train_mask_paths,
            self.val_mask_paths,
        ) = train_test_split(images_paths, masks_paths, test_size=0.2, random_state=42)

    def get_train_val_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = CILDataset(
            self.train_image_paths,
            self.train_mask_paths,
            PATCH_SIZE,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=self.augment,
        )

        val_dataset = CILDataset(
            self.val_image_paths,
            self.val_mask_paths,
            PATCH_SIZE,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=None,
        )

        train_dataloader = DataLoader(train_dataset, self.batch_size, self.shuffle)
        val_dataloader = DataLoader(val_dataset, self.batch_size, self.shuffle)

        return train_dataloader, val_dataloader


@DATASET_REGISTRY.register("cil")
class CILDataset(Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(
        self,
        image_paths,
        mask_paths,
        patch_size,
        cutoff,
        device,
        resize_to=(400, 400),
        augment=None,
    ):
        if augment is None:
            augment = []
        self.items = list(zip(image_paths, mask_paths))
        self.patch_size = patch_size
        self.cutoff = cutoff
        self.device = device
        self.resize_to = resize_to
        self.augment = augment

        self.geometric_transform = A.Compose(
            [
                A.Rotate(30),
                A.VerticalFlip(),
                A.VerticalFlip(),
            ]
        )

        self.color_transform = A.Compose(
            [
                A.ColorJitter(brightness=0.2, contrast=0.2),
                A.AdvancedBlur(blur_limit=(5, 9), sigma_x_limit=(0.1, 5), sigma_y_limit=(0.1, 5)),
            ]
        )

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        if "geometric" in self.augment:
            z = self.geometric_transform(image=x, mask=y)
            x = z["image"]
            y = z["mask"]

        if "color" in self.augment:
            x = self.color_transform(image=x)["image"]

        # cv2.imshow('x', x)
        # cv2.imshow('y', y)
        # cv2.waitKey(0)
        return x, y

    def __getitem__(self, item):
        # return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))
        img_path, mask_path = self.items[item]

        image = np.array(Image.open(img_path))[:, :, :3].astype(np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0

        if self.resize_to != (image.shape[0], image.shape[1]):  # resize images
            image = cv2.resize(image, dsize=self.resize_to)
            mask = cv2.resize(mask, dsize=self.resize_to)

        image, mask = self._preprocess(image, mask)

        image = np.moveaxis(
            image, -1, 0
        )  # pytorch works with CHW format instead of HWC
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        mask = np.moveaxis(mask, -1, 0)  # pytorch works with CHW format instead of HWC

        image_tensor = np_to_tensor(image, self.device)
        mask_tensor = np_to_tensor(mask, self.device)

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.items)
