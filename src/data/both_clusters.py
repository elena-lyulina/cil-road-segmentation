from glob import glob
from typing import Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from src.constants import DATA_PATH, DEVICE, CUTOFF, PATCH_SIZE
from src.data.DeepGlobe_dataset import DeepGlobeDataset
from src.data.ThirtyK_dataset import ThirtyKDataset
from src.data.cluster0 import Cluster0Dataset
from src.data.cluster1 import Cluster1Dataset
from src.data.datahandler import DATAHANDLER_REGISTRY, DataHandler
from src.data.utils import DATASET_REGISTRY, np_to_tensor
from src.data.NinetyK_dataset import NinetyKDataset


@DATAHANDLER_REGISTRY.register("both_clusters")
class AllDataHandler(DataHandler):
    dataset_path = DATA_PATH.joinpath("clusters")

    dataset_path_cluster0 = dataset_path.joinpath("cluster0")
    images_path_cluster0 = dataset_path_cluster0.joinpath("images")
    masks_path_cluster0= dataset_path_cluster0.joinpath("masks")

    dataset_path_cluster1 = dataset_path.joinpath("cluster1")
    images_path_cluster1 = dataset_path_cluster1.joinpath("images")
    masks_path_cluster1 = dataset_path_cluster1.joinpath("masks")



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

        images_paths_cluster0 = [f for f in sorted(glob(str(self.images_path_cluster0) + "/*.png"))]
        masks_paths_cluster0= [f for f in sorted(glob(str(self.masks_path_cluster0) + "/*.png"))]

        images_paths_cluster1 = [f for f in sorted(glob(str(self.images_path_cluster1) + "/*.png"))]
        masks_paths_cluster1 = [f for f in sorted(glob(str(self.masks_path_cluster1) + "/*.png"))]

        (
            self.train_image_paths_cluster0,
            self.val_image_paths_cluster0,
            self.train_mask_paths_cluster0,
            self.val_mask_paths_cluster0,
        ) = train_test_split(images_paths_cluster0, masks_paths_cluster0, test_size=0.2, random_state=42)

        (
            self.train_image_paths_cluster1,
            self.val_image_paths_cluster1,
            self.train_mask_paths_cluster1,
            self.val_mask_paths_cluster1,
        ) = train_test_split(images_paths_cluster1, masks_paths_cluster1, test_size=0.2, random_state=42)


    def get_train_val_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset_cluster0 = Cluster0Dataset(
            self.train_image_paths_cluster0,
            self.train_mask_paths_cluster0,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=self.augment,
            masking_params=self.masking_params,
        )

        val_dataset_cluster0 = Cluster0Dataset(
            self.val_image_paths_cluster0,
            self.val_mask_paths_cluster0,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=["masked"] if "masked" in self.augment else None,
            masking_params=self.masking_params,
        )

        train_dataset_cluster1 = Cluster1Dataset(
            self.train_image_paths_cluster1,
            self.train_mask_paths_cluster1,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=self.augment,
            masking_params=self.masking_params,
        )

        val_dataset_cluster1 = Cluster1Dataset(
            self.val_image_paths_cluster1,
            self.val_mask_paths_cluster1,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=["masked"] if "masked" in self.augment else None,
            masking_params=self.masking_params,
        )



        train_dataset = ConcatDataset([train_dataset_cluster0, train_dataset_cluster1])
        val_dataset = ConcatDataset([val_dataset_cluster0, val_dataset_cluster1])

        train_dataloader = DataLoader(train_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, prefetch_factor=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, prefetch_factor=4, pin_memory=True)

        return train_dataloader, val_dataloader

