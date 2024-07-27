import os
from glob import glob
from pathlib import Path
from typing import Tuple, List

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


def end2end(config_paths: List[Path], voter: str, resize_to: Tuple[int, int] = (400, 400)):
    # If CIL dataset was used in the config, gets the image sizes from there
    # Then loads the model and runs it on test images, resized to the found size if possible

    assert (len(config_paths) == 6)
    submissions = []
    model_config_paths = config_paths[:5]
    for config_path in model_config_paths:
        experiment_name = get_experiment_name_from_config(config_path)
        submission_name = config_path.stem

        experiment_name = 'None'  # Doesn't work if weights are moved

        config = load_config(config_path)
        if config['dataset']['name'] == 'cil':
            real_resize_to = config['dataset']['params'].get('resize_to')
            if real_resize_to is not None:
                print(
                    f"Found the size of the CIL images the model was trained on: {real_resize_to}")
                if resize_to != real_resize_to:
                    print(
                        f"Using the found size {real_resize_to} instead of given {resize_to}")
                    resize_to = real_resize_to

        model_path = get_model_path_from_config(config_path)
        model, _ = load_checkpoint(model_path)
        model.eval()

        submissions.append(test_model_on_full_images_for_end2end(
            model, experiment_name, submission_name, resize_to))

    if voter == "hard_pixel":
        interim_predictions = hard_voting_pixel_level(submissions)
    elif voter == "soft_pixel":
        interim_predictions = soft_voting_pixel_level(submissions)
    elif voter == "hard_patch":
        interim_predictions = hard_voting_patch_level(submissions)
    elif voter == "soft_patch":
        interim_predictions = soft_voting_patch_level(submissions)
    else:
        raise ValueError("Invalid voter type")

    mae_config_path = config_paths[5]
    experiment_name = get_experiment_name_from_config(mae_config_path)
    submission_name = mae_config_path.stem
    mae_model_path = get_model_path_from_config(mae_config_path)
    model, _ = load_checkpoint(mae_model_path)
    model.eval()
    run_mae(interim_predictions, model, experiment_name, submission_name)


def run_mae(voter_op, model: nn.Module, experiment_name: str, submission_name: str):

    experiment_prediction_path = PREDICTION_PATH.joinpath(experiment_name)
    os.makedirs(experiment_prediction_path, exist_ok=True)
    prediction_file_path = experiment_prediction_path.joinpath(
        f"{submission_name}.csv")
    prediction_file_path = make_sure_unique([prediction_file_path])[0]
    original_size = test_images.shape[1:3]
    # test_paths = TEST_IMAGES_PATH.glob('*.png')
    test_paths = glob(str(TEST_IMAGES_PATH) + '/*.png')

    test_images = np_to_tensor(np.moveaxis(
        voter_op, -1, 1), DEVICE)  # shape (144, 1, H, W)
    print("Running the MAE model on test images..")
    test_pred = [model(t).detach().cpu().numpy()
                 for t in tqdm(test_images.unsqueeze(1))]
    test_pred = np.concatenate(test_pred, 0)  # shape (144, 1, H, W)
    # shape (144, H, W, 1), CxHxW to HxWxC
    test_pred = np.moveaxis(test_pred, 1, -1)
    # shape (144, 400, 400, 1), resize back to original shape
    test_pred = np.stack([cv2.resize(img, dsize=original_size)
                         for img in test_pred], 0)
    # now split into patches and compute labels
    # shape (144, 25, 16, 25, 16)
    test_pred = test_pred.reshape(
        (-1, original_size[0] // PATCH_SIZE, PATCH_SIZE, original_size[0] // PATCH_SIZE, PATCH_SIZE))
    test_pred = np.moveaxis(test_pred, 2, 3)  # shape (144, 25, 25, 16, 16)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) >
                         CUTOFF)  # shape (144, 25, 25)

    print(f"Creating submission file {prediction_file_path}")
    create_submission(test_pred, test_paths, prediction_file_path, PATCH_SIZE)


def test_model_on_full_images_for_end2end(model: nn.Module, experiment_name: str, submission_name: str, resize_to: Tuple[int, int] = (400, 400)):
    # Runs the model on test images (possibly resized to a required size),
    # resizes the model's output back to the original size,
    # then creates patches, calculates the cutoff, and creates a submission file

    test_images = load_all_from_path(
        str(TEST_IMAGES_PATH))  # shape (144, 400, 400, 4)

    original_size = test_images.shape[1:3]  # shape (400, 400)
    # shape (144, H, W, 4), resize if needed for the model
    test_images = np.stack([cv2.resize(img, dsize=resize_to)
                           for img in test_images], 0)
    # shape (144, H, W, 3), leave only 3 channels
    test_images = test_images[:, :, :, :3]
    test_images = np_to_tensor(np.moveaxis(
        test_images, -1, 1), DEVICE)  # shape (144, 3, H, W)

    print("Running the model on test images..")
    test_pred = [model(t).detach().cpu().numpy()
                 for t in tqdm(test_images.unsqueeze(1))]
    test_pred = np.concatenate(test_pred, 0)  # shape (144, 1, H, W)
    # shape (144, H, W, 1), CxHxW to HxWxC
    test_pred = np.moveaxis(test_pred, 1, -1)
    # shape (144, 400, 400, 1), resize back to original shape
    test_pred = np.stack([cv2.resize(img, dsize=original_size)
                         for img in test_pred], 0)

    return test_pred


def hard_voting_pixel_level(all_model_outputs):
    all_predictions = np.zeros_like(all_model_outputs[0])

    for model_output in all_model_outputs:
        all_predictions += (model_output > 0.5).astype(np.float32)

    # Majority vote
    all_predictions = (all_predictions >= (
        len(all_model_outputs) / 2)).astype(np.float32)

    return all_predictions


def soft_voting_pixel_level(all_model_outputs):
    all_predictions = np.zeros_like(all_model_outputs[0], dtype=np.float32)

    for model_output in all_model_outputs:
        all_predictions += model_output

    # Average predictions
    all_predictions = (all_predictions / len(all_model_outputs)) > 0.5
    all_predictions = all_predictions.astype(np.float32)

    return all_predictions


def hard_voting_patch_level(all_model_outputs):
    all_predictions = np.zeros_like(all_model_outputs[0])

    for model_output in all_model_outputs:
        all_predictions += (model_output > 0.5).astype(np.float32)

    # Reshape to patch level
    all_predictions_patches = all_predictions.reshape(
        all_predictions.shape[0], 25, 16, 25, 16)
    all_predictions_patches = all_predictions_patches.sum(
        axis=(2, 4)) >= (0.25 * 256 * len(all_model_outputs))
    all_predictions_patches = all_predictions_patches.astype(np.float32)

    return all_predictions_patches.reshape(all_predictions.shape)


def soft_voting_patch_level(all_model_outputs):
    all_predictions = np.zeros_like(all_model_outputs[0], dtype=np.float32)

    for model_output in all_model_outputs:
        all_predictions += model_output

    # Reshape to patch level
    all_predictions_patches = all_predictions.reshape(
        all_predictions.shape[0], 25, 16, 25, 16)
    all_predictions_patches = all_predictions_patches.sum(
        axis=(2, 4)) >= (0.25 * 256 * len(all_model_outputs))
    all_predictions_patches = all_predictions_patches.astype(np.float32)

    return all_predictions_patches.reshape(all_predictions.shape)
