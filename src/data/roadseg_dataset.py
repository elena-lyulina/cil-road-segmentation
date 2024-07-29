import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import random

from src.data.utils import np_to_tensor


class RoadSegDataset(Dataset):
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
            "noise_threshold": 0.5,
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
                A.AdvancedBlur(
                    blur_limit=(5, 9), sigma_x_limit=(0.1, 5), sigma_y_limit=(0.1, 5)
                ),
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
            x = self.apply_masking(y.copy())

        # cv2.imwrite('x.png', x)
        # cv2.imwrite('y.png', y)

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
        noise_threshold = self.masking_params.get("noise_threshold", 0.5)

        # Apply patch flipping
        for _ in range(num_flip_patches):
            i, j = random.randint(0, 400 - flip_patch_size), random.randint(0, 400 - flip_patch_size)
            mask[i : i + flip_patch_size, j : j + flip_patch_size] = 1 - mask[i : i + flip_patch_size, j : j + flip_patch_size]
        
        # Apply patch masking
        for _ in range(num_zero_patches):
            i, j = random.randint(0, 400 - zero_patch_size), random.randint(0, 400 - zero_patch_size)
            mask[i : i + zero_patch_size, j : j + zero_patch_size] = 0

        # Apply Gaussian noise and binarize it
        if noise_threshold < 100:
            noise = np.random.normal(0, 1, mask.shape)
            random_ones = (noise > noise_threshold).astype(np.float32)

            noise = np.random.normal(0, 1, mask.shape)
            random_zeros = (noise > noise_threshold).astype(np.float32)
        
            mask = mask + random_ones - random_zeros
            mask = np.clip(mask, 0, 1)

        return mask

    def __getitem__(self, item):
        # return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))
        img_path, mask_path = self.items[item]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)[:, :, :3].astype(np.float32) / 255.0
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

        image_tensor = np_to_tensor(image, 'cpu')
        mask_tensor = np_to_tensor(mask, 'cpu')

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.items)
