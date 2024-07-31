import os
from glob import glob
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from torch import nn
from tqdm import tqdm

from src.constants import OUT_PATH, DATA_PATH, PATCH_SIZE, CUTOFF, DEVICE
from src.data.utils import load_all_from_path, np_to_tensor
from src.experiments.config import get_model_path_from_config, get_experiment_name_from_config, load_config
from src.submission.mask_to_submission import create_submission
from src.train.train import load_checkpoint, make_sure_unique

TEST_IMAGES_PATH = DATA_PATH.joinpath('cil/test/images/')
PREDICTION_PATH = OUT_PATH.joinpath('predictions/')


def test_on_full_images(config_path: Path, resize_to: Tuple[int, int] = (400, 400)):
    # If CIL dataset was used in the config, gets the image sizes from there
    # Then loads the model and runs it on test images, resized to the found size if possible

    experiment_name = get_experiment_name_from_config(config_path)
    submission_name = config_path.stem

    experiment_name = 'None' # Doesn't work if weights are moved

    config = load_config(config_path)
    resize_to = get_cil_resize_param(config, resize_to)
    model_path = get_model_path_from_config(config_path)
    model, _ = load_checkpoint(model_path)
    model.eval()

    test_model_on_full_images(model, experiment_name, submission_name, resize_to)


def get_cil_resize_param(config, resize_to):
    if config['dataset']['name'] == 'cil':
        real_resize_to = config['dataset']['params'].get('resize_to')
        if real_resize_to is not None:
            print(f"Found the size of the CIL images the model was trained on: {real_resize_to}")
            if resize_to != real_resize_to:
                print(f"Using the found size {real_resize_to} instead of given {resize_to}")
                resize_to = real_resize_to
    return resize_to


def test_model_on_full_images(model: nn.Module, experiment_name: str, submission_name: str, resize_to: Tuple[int, int] = (400, 400)):
    # Runs the model on test images (possibly resized to a required size),
    # resizes the model's output back to the original size,
    # then creates patches, calculates the cutoff, and creates a submission file
    experiment_prediction_path = PREDICTION_PATH.joinpath(experiment_name)
    os.makedirs(experiment_prediction_path, exist_ok=True)
    prediction_file_path = experiment_prediction_path.joinpath(f"{submission_name}.csv")
    prediction_file_path = make_sure_unique([prediction_file_path])[0]

    # test_paths = TEST_IMAGES_PATH.glob('*.png')
    test_paths = glob(str(TEST_IMAGES_PATH) + '/*.png')
    test_images = load_all_from_path(str(TEST_IMAGES_PATH)) # shape (144, 400, 400, 4)

    original_size = test_images.shape[1:3] # shape (400, 400)
    test_images = np.stack([cv2.resize(img, dsize=resize_to) for img in test_images], 0) # shape (144, H, W, 4), resize if needed for the model
    test_images = test_images[:, :, :, :3] # shape (144, H, W, 3), leave only 3 channels
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), DEVICE) # shape (144, 3, H, W)

    print("Running the model on test images..")
    test_pred = [model(t).detach().cpu().numpy() for t in tqdm(test_images.unsqueeze(1))]
    test_pred = np.concatenate(test_pred, 0) # shape (144, 1, H, W)
    test_pred = np.moveaxis(test_pred, 1, -1)  # shape (144, H, W, 1), CxHxW to HxWxC
    test_pred = np.stack([cv2.resize(img, dsize=original_size) for img in test_pred], 0)  # shape (144, 400, 400, 1), resize back to original shape

    # now split into patches and compute labels
    test_pred = test_pred.reshape((-1, original_size[0] // PATCH_SIZE, PATCH_SIZE, original_size[0] // PATCH_SIZE, PATCH_SIZE)) # shape (144, 25, 16, 25, 16)
    test_pred = np.moveaxis(test_pred, 2, 3)  # shape (144, 25, 25, 16, 16)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF) # shape (144, 25, 25)

    print(f"Creating submission file {prediction_file_path}")
    create_submission(test_pred, test_paths, prediction_file_path, PATCH_SIZE)
