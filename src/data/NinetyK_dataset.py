from glob import glob
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
import torch
import random

from src.constants import DATA_PATH, DEVICE, CUTOFF, PATCH_SIZE
from src.data.datahandler import DATAHANDLER_REGISTRY, DataHandler
from src.data.utils import DATASET_REGISTRY, np_to_tensor


@DATAHANDLER_REGISTRY.register("90k")
class NinetyKDataHandler(DataHandler):
    dataset_path = DATA_PATH.joinpath("90k")

    train_images_path = dataset_path.joinpath("images")
    train_masks_path = dataset_path.joinpath("masks")

    def __init__(self, batch_size=4, num_workers=4, shuffle=True, resize_to=(400, 400), augment=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        train_dataset = NinetyKDataset(
            self.train_image_paths,
            self.train_mask_paths,
            PATCH_SIZE,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=self.augment,
        )

        val_dataset = NinetyKDataset(
            self.val_image_paths,
            self.val_mask_paths,
            PATCH_SIZE,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=["masked"] if "masked" in self.augment else None,
        )

        train_dataloader = DataLoader(train_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, prefetch_factor=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, prefetch_factor=4, pin_memory=True)

        return train_dataloader, val_dataloader


@DATASET_REGISTRY.register("90k")
class NinetyKDataset(Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(
        self,
        image_paths,
        mask_paths,
        cutoff,
        device,
        resize_to=(400, 400),
        augment=None,
        masking_params = {
            "num_zero_patches": 8,
            "zero_patch_size": 50,
            "num_flip_patches": 25,
            "flip_patch_size": 16,
        }
    ):
        if augment is None:
            augment = []
        self.items = list(zip(image_paths, mask_paths))
        self.cutoff = cutoff
        self.device = device
        self.resize_to = resize_to
        self.augment = augment
        self.masking_params = masking_params

        self.geometric_transform = A.Compose(
            [
                A.RandomRotate90(),
                A.VerticalFlip(),
                A.HorizontalFlip(),
            ]
        )

        self.color_transform = A.Compose(
            [
                # params should be a range, we might want to look into it later to tune these params or use defaults
                A.ColorJitter(brightness=(0.2, 0.2), contrast=(0.2, 0.2)),
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

        if "masked" in self.augment:
            x = self.apply_masking(y)

        # cv2.imshow('x', x)
        # cv2.imshow('y', y)
        # cv2.waitKey(0)
        return x, y

    def apply_masking(self, mask):
        #read masking params into local variables, check if they are present in masking_params
        num_zero_patches = self.masking_params.get("num_zero_patches", 8)
        zero_patch_size = self.masking_params.get("zero_patch_size", 50)
        num_flip_patches = self.masking_params.get("num_flip_patches", 25)
        flip_patch_size = self.masking_params.get("flip_patch_size", 16)

        # Apply patch flipping
        for _ in range(num_flip_patches):
            i, j = random.randint(0, 400 - flip_patch_size), random.randint(0, 400 - flip_patch_size)
            mask[i : i + flip_patch_size, j : j + flip_patch_size] = 1 - mask[i : i + flip_patch_size, j : j + flip_patch_size]
        
        # Apply patch masking
        for _ in range(num_zero_patches):
            i, j = random.randint(0, 400 - zero_patch_size), random.randint(0, 400 - zero_patch_size)
            mask[i : i + zero_patch_size, j : j + zero_patch_size] = 0

        return mask

    def __getitem__(self, item):
        # return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))
        img_path, mask_path = self.items[item]
        image = np.array(Image.open(img_path))[:, :, :3].astype(np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0

        if self.resize_to != (image.shape[0], image.shape[1]):  # resize images
            image = cv2.resize(image, dsize=self.resize_to)
            mask = cv2.resize(mask, dsize=self.resize_to)

        image, mask = self._preprocess(image, mask)
        if "masked" in self.augment:
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        image = np.moveaxis(
            image, -1, 0
        )  # pytorch works with CHW format instead of HWC
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        mask = np.moveaxis(mask, -1, 0)  # pytorch works with CHW format instead of HWC

        image_tensor = np_to_tensor(image, "cpu")
        mask_tensor = np_to_tensor(mask, "cpu")

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.items)
