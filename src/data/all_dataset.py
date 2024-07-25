from glob import glob
from typing import Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from src.constants import DATA_PATH, DEVICE, CUTOFF, PATCH_SIZE
from src.data.DeepGlobe_dataset import DeepGlobeDataset
from src.data.ThirtyK_dataset import ThirtyKDataset
from src.data.datahandler import DATAHANDLER_REGISTRY, DataHandler
from src.data.utils import DATASET_REGISTRY, np_to_tensor
from src.data.NinetyK_dataset import NinetyKDataset


@DATAHANDLER_REGISTRY.register("all")
class AllDataHandler(DataHandler):
    dataset_path_90k = DATA_PATH.joinpath("90k")

    images_path_90k = dataset_path_90k.joinpath("images")
    masks_path_90k = dataset_path_90k.joinpath("masks")

    dataset_path_30k = DATA_PATH.joinpath("30k")

    images_path_30k = dataset_path_30k.joinpath("images")
    masks_path_30k = dataset_path_30k.joinpath("masks")

    dataset_path_DeepGlobe = DATA_PATH.joinpath("DeepGlobe")

    images_path_DeepGlobe = dataset_path_DeepGlobe.joinpath("images")
    masks_path_DeepGlobe = dataset_path_DeepGlobe.joinpath("masks")

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
        self.augment = augment
        self.masking_params = masking_params

        images_paths_90k = [f for f in sorted(glob(str(self.images_path_90k) + "/*.png"))]
        masks_paths_90k= [f for f in sorted(glob(str(self.masks_path_90k) + "/*.png"))]

        images_paths_30k = [f for f in sorted(glob(str(self.images_path_30k) + "/*.png"))]
        masks_paths_30k = [f for f in sorted(glob(str(self.masks_path_30k) + "/*.png"))]

        images_paths_DeepGlobe = [f for f in sorted(glob(str(self.images_path_DeepGlobe) + "/*.png"))]
        masks_paths_DeepGlobe= [f for f in sorted(glob(str(self.masks_path_DeepGlobe) + "/*.png"))]

        (
            self.train_image_paths_90k,
            self.val_image_paths_90k,
            self.train_mask_paths_90k,
            self.val_mask_paths_90k,
        ) = train_test_split(images_paths_90k, masks_paths_90k, test_size=0.2, random_state=42)

        (
            self.train_image_paths_30k,
            self.val_image_paths_30k,
            self.train_mask_paths_30k,
            self.val_mask_paths_30k,
        ) = train_test_split(images_paths_30k, masks_paths_30k, test_size=0.2, random_state=42)

        (
            self.train_image_paths_DeepGlobe,
            self.val_image_paths_DeepGlobe,
            self.train_mask_paths_DeepGlobe,
            self.val_mask_paths_DeepGlobe,
        ) = train_test_split(images_paths_DeepGlobe, masks_paths_DeepGlobe, test_size=0.2, random_state=42)

    def get_train_val_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset_90k = NinetyKDataset(
            self.train_image_paths_90k,
            self.train_mask_paths_90k,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=self.augment,
            masking_params=self.masking_params,
        )

        val_dataset_90k = NinetyKDataset(
            self.val_image_paths_90k,
            self.val_mask_paths_90k,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=["masked"] if "masked" in self.augment else None,
            masking_params=self.masking_params,
        )

        train_dataset_30k = ThirtyKDataset(
            self.train_image_paths_30k,
            self.train_mask_paths_30k,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=self.augment,
            masking_params=self.masking_params
        )

        val_dataset_30k = ThirtyKDataset(
            self.val_image_paths_30k,
            self.val_mask_paths_30k,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=["masked"] if "masked" in self.augment else None,
            masking_params=self.masking_params
        )

        train_dataset_DeepGlobe = DeepGlobeDataset(
            self.train_image_paths_DeepGlobe,
            self.train_mask_paths_DeepGlobe,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=self.augment,
            masking_params=self.masking_params
        )

        val_dataset_DeepGlobe = DeepGlobeDataset(
            self.val_image_paths_DeepGlobe,
            self.val_mask_paths_DeepGlobe,
            CUTOFF,
            DEVICE,
            resize_to=self.resize_to,
            augment=["masked"] if "masked" in self.augment else None,
            masking_params=self.masking_params
        )

        train_dataset = ConcatDataset([train_dataset_90k, train_dataset_30k, train_dataset_DeepGlobe])
        val_dataset = ConcatDataset([val_dataset_90k, val_dataset_30k, val_dataset_DeepGlobe])

        train_dataloader = DataLoader(train_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, prefetch_factor=4, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, self.batch_size, self.shuffle, num_workers=self.num_workers, prefetch_factor=4, pin_memory=True)

        return train_dataloader, val_dataloader

