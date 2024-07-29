from glob import glob
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A

from src.data.roadseg_dataset import RoadSegDataset

from src.constants import DATA_PATH, DEVICE, CUTOFF, PATCH_SIZE
from src.data.datahandler import DATAHANDLER_REGISTRY, DataHandler
from src.data.utils import DATASET_REGISTRY, np_to_tensor


@DATAHANDLER_REGISTRY.register("cluster1")
class Cluster1DataHandler(DataHandler):
    dataset_path = DATA_PATH.joinpath("clusters")
    dataset_path = dataset_path.joinpath("cluster1")

    train_images_path = dataset_path.joinpath("images")
    train_masks_path = dataset_path.joinpath("masks")

    def __init__(self, batch_size=4, num_workers=4, shuffle=True, resize_to=(400, 400), augment=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.resize_to = resize_to
        self.augment = augment if augment else []

        images_paths = [f for f in sorted(glob(str(self.train_images_path) + "/*.png"))]
        masks_paths = [f for f in sorted(glob(str(self.train_masks_path) + "/*.png"))]

        (
            self.train_image_paths,
            self.val_image_paths,
            self.train_mask_paths,
            self.val_mask_paths,
        ) = train_test_split(images_paths, masks_paths, test_size=0.2, random_state=42)

    def get_train_val_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = Cluster1Dataset(
            self.train_image_paths,
            self.train_mask_paths,
            PATCH_SIZE,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=self.augment,
        )

        val_dataset = Cluster1Dataset(
            self.val_image_paths,
            self.val_mask_paths,
            PATCH_SIZE,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=["masked"] if "masked" in self.augment else None,
        )


        train_dataloader = DataLoader(train_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, drop_last=True, prefetch_factor=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, drop_last=True, prefetch_factor=4, pin_memory=True)

        return train_dataloader, val_dataloader


@DATASET_REGISTRY.register("cluster1")
class Cluster1Dataset(RoadSegDataset):

    def __getitem__(self, item):
        cluster_id = np_to_tensor(np.array([1]), 'cpu')
        image_tensor, mask_tensor = super().__getitem__(item)

        return image_tensor, mask_tensor, cluster_id
