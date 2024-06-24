import os
from glob import glob
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.constants import DATA_PATH
from src.data.datahandler import DATAHANDLER_REGISTRY, DataHandler
from src.data.utils import DATASET_REGISTRY


# DISCLAIMER: I just copied it from the Elouan's code (what was in massachusetts_pretraining.py) so no guarantees, didn't run it
@DATAHANDLER_REGISTRY.register("massachusetts")
class MassachusettsDataHandler(DataHandler):
    dataset_path = DATA_PATH.joinpath("massachusetts")
    train_images_path = dataset_path.joinpath("train")
    train_masks_path = dataset_path.joinpath("train_labels")

    test_images_path = dataset_path.joinpath("test")
    test_masks_path = dataset_path.joinpath("test_labels")

    def __init__(self, batch_size=32, num_workers=16, pin_memory=True, shuffle=True):
        # num_workers: tune that to CPU capacities, keep high number of parallelization used
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

    def get_train_val_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = MassachusettsDataset(self.train_images_path, self.train_masks_path)
        val_dataset = MassachusettsDataset(self.test_images_path, self.test_masks_path)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                       num_workers=self.num_workers,
                                                       pin_memory=self.pin_memory, shuffle=self.shuffle)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,
                                                     num_workers=self.num_workers,
                                                     pin_memory=self.pin_memory, shuffle=self.shuffle)

        return train_dataloader, val_dataloader


@DATASET_REGISTRY.register("massachusetts")
class MassachusettsDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.tiff')))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.tif')))

        self.crop_size = 384
        self.image_size = 1500
        self.indices = []

        ### Allows 9 patches 384x384 for each 1500x1500 original to be used, only compute once the double loop
        for i in range(len(self.image_paths)):
            for y in range(0, self.image_size, self.crop_size):
                for x in range(0, self.image_size, self.crop_size):
                    if y + self.crop_size <= self.image_size and x + self.crop_size <= self.image_size:
                        self.indices.append((i, x, y))

    def __len__(self):
        return len(self.indices)  ### allows all crops to be used without change to dataloading or batch handling

    def __getitem__(self, idx):
        image_idx, x, y = self.indices[idx]
        img_path = self.image_paths[image_idx]
        mask_path = self.mask_paths[image_idx]

        image = np.array(Image.open(img_path), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.float32)

        image /= 255.0
        mask /= 255.0

        # crop to 384x384 as used in main dataset
        image_crop = image[y:y + self.crop_size, x:x + self.crop_size, :]
        mask_crop = mask[y:y + self.crop_size, x:x + self.crop_size]

        #CHW for torch
        image_crop = np.transpose(image_crop, (2, 0, 1))

        return torch.from_numpy(image_crop), torch.from_numpy(mask_crop)
