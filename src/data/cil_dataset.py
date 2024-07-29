import json
from glob import glob
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import albumentations as A
import random

from src.data.roadseg_dataset import RoadSegDataset

from src.constants import DATA_PATH, DEVICE, CUTOFF, PATCH_SIZE
from src.data.datahandler import DATAHANDLER_REGISTRY, DataHandler
from src.data.utils import DATASET_REGISTRY, np_to_tensor
from src.data.roadseg_dataset import RoadSegDataset


@DATAHANDLER_REGISTRY.register("cil")
class CILDataHandler(DataHandler):
    dataset_path = DATA_PATH.joinpath("cil")

    train_path = dataset_path.joinpath("training")
    train_images_path = train_path.joinpath("images")
    train_masks_path = train_path.joinpath("groundtruth")

    def __init__(
        self,
        batch_size=4,
        num_workers=4,
        shuffle=True,
        resize_to=(400, 400),
        augment=None,
        masking_params = {
            "num_zero_patches": 8,
            "zero_patch_size": 50,
            "num_flip_patches": 25,
            "flip_patch_size": 16,
        }
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.resize_to = resize_to
        self.augment = augment if augment else []
        self.masking_params = masking_params

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
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=self.augment,
            masking_params=self.masking_params,
        )

        val_dataset = CILDataset(
            self.val_image_paths,
            self.val_mask_paths,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=["masked"] if "masked" in self.augment else None,
            masking_params=self.masking_params,
        )


        train_dataloader = DataLoader(train_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, drop_last=True, prefetch_factor=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, drop_last=True, prefetch_factor=4, pin_memory=True)

        return train_dataloader, val_dataloader


class CILDataset(RoadSegDataset):
    
    def __getitem__(self, item):
        path = DATA_PATH.joinpath("cil")
        self.cluster_dict = json.load(open(path.joinpath('CLIP_clusters.json')))
        img_path, _ = self.items[item]

        name = 'cil/training/images/' + Path(img_path).name
        cluster_id = np_to_tensor(np.array([self.cluster_dict[name]]), 'cpu')
        
        image_tensor, mask_tensor = super().__getitem__(item)

        return image_tensor, mask_tensor, cluster_id

