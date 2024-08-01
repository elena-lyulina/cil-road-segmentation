import json
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
from src.voter import hard_voting_pixel_level, soft_voting_pixel_level, hard_voting_patch_level, soft_voting_patch_level

TEST_IMAGES_PATH = DATA_PATH.joinpath('cil/test/images/')
PREDICTION_PATH = OUT_PATH.joinpath('predictions/')


def end2end(config_paths: List[Path], voter: str, experiment_name, with_mae=True, cluster=False, mae_config_path=None):
    # If CIL dataset was used in the config, gets the image sizes from there
    # Then loads the model and runs it on test images, resized to the found size if possible

    submissions = []
    for config_path in config_paths:

        if not cluster:
            submission_name = config_path.stem
            model_path = get_model_path_from_config(config_path)
            model, _ = load_checkpoint(model_path)
            model.eval()

            submissions.append(test_model_on_full_images_for_end2end(model0=model, cluster=cluster))

        else:
            config0 = config_path[0]
            config1 = config_path[1]
            submission_name = config0.stem

            model0_path = get_model_path_from_config(config0)
            model1_path = get_model_path_from_config(config1)
            model0, _ = load_checkpoint(model0_path)
            model1, _ = load_checkpoint(model1_path)
            model0.eval()
            model1.eval()

            submissions.append(test_model_on_full_images_for_end2end(model0=model0, model1=model1, cluster=cluster, mae_config_path=mae_config_path))

    if voter == "hard_pixel":
        interim_predictions = hard_voting_pixel_level(submissions)
    elif voter == "soft_pixel":
        interim_predictions = soft_voting_pixel_level(submissions)
    elif voter == "hard_patch":
        interim_predictions = np.expand_dims(hard_voting_patch_level(submissions), 1)
    elif voter == "soft_patch":
        interim_predictions = np.expand_dims(soft_voting_patch_level(submissions), 1)
    elif voter == "None":
        interim_predictions = submissions
    else:
        raise ValueError("Invalid voter type")

    if with_mae:
        mae_model_path = get_model_path_from_config(mae_config_path)
        model, _ = load_checkpoint(mae_model_path)
        model.eval()

        test_images = np_to_tensor(np.moveaxis(
            interim_predictions, -1, 1), DEVICE)  # shape (144, 1, H, W)

        pred = [model(t)[0].detach().cpu().numpy()
                     for t in tqdm(test_images.unsqueeze(1))]

        pred = np.concatenate(pred, 0)  # shape (144, 1, H, W)
        # shape (144, H, W, 1), CxHxW to HxWxC
        pred = np.moveaxis(pred, 1, -1)

        interim_predictions = (pred > 0.5).astype(np.float32)

    experiment_prediction_path = PREDICTION_PATH.joinpath(experiment_name)
    os.makedirs(experiment_prediction_path, exist_ok=True)
    prediction_file_path = experiment_prediction_path.joinpath(
        f"{submission_name}.csv")
    prediction_file_path = make_sure_unique([prediction_file_path])[0]
    # test_paths = TEST_IMAGES_PATH.glob('*.png')
    test_paths = glob(str(TEST_IMAGES_PATH) + '/*.png')

    # shape (144, 400, 400, 1), resize back to original shape
    pred = np.stack([img for img in interim_predictions], 0)

    pred = pred.reshape(
        (-1, 400 // PATCH_SIZE, PATCH_SIZE, 400 // PATCH_SIZE, PATCH_SIZE))
    pred = np.moveaxis(pred, 2, 3)  # shape (144, 25, 25, 16, 16)
    pred = np.round(np.mean(pred, (-1, -2)) >
                         CUTOFF)  # shape (144, 25, 25)

    print(f"Creating submission file {prediction_file_path}")
    create_submission(pred, test_paths, prediction_file_path, PATCH_SIZE)


def test_model_on_full_images_for_end2end(model0: nn.Module, model1=None, cluster=False, mae_config_path=None):
    # Runs the model on test images (possibly resized to a required size),
    # resizes the model's output back to the original size,
    # then creates patches, calculates the cutoff, and creates a submission file

    test_images = load_all_from_path(
        str(TEST_IMAGES_PATH))  # shape (144, 400, 400, 4)

    mae_model_path = get_model_path_from_config(mae_config_path)
    model, _ = load_checkpoint(mae_model_path)
    model.eval()

    # test_images = np.stack([img for img in test_images], 0)
    # shape (144, H, W, 3), leave only 3 channels
    test_images = test_images[:, :, :, :3]
    test_images = np_to_tensor(np.moveaxis(
        test_images, -1, 1), DEVICE)  # shape (144, 3, H, W)

    print("Running the model on test images..")

    if not cluster:
        test_pred = [model0(t).detach().cpu().numpy()
                     for t in tqdm(test_images.unsqueeze(1))]
    else:
        path = DATA_PATH.joinpath("cil")
        cluster_dict = json.load(open(path.joinpath('CLIP_clusters.json')))

        names = [Path(f).name for f in sorted(glob(str(TEST_IMAGES_PATH) + '/*.png'))]

        test_pred = []

        for i, t in tqdm(enumerate(test_images.unsqueeze(1)), total=len(names)):
            name = 'cil/test/images/' + names[i]
            cluster_id = np_to_tensor(np.array([cluster_dict[name]]), 'cpu')
            pred = model0(t) if cluster_id == 0 else model1(t) if cluster_id == 1 else None
            test_pred.append(pred.detach().cpu().numpy())

    test_pred = np.concatenate(test_pred, 0)  # shape (144, 1, H, W)
    # shape (144, H, W, 1), CxHxW to HxWxC
    test_pred = np.moveaxis(test_pred, 1, -1)
    # shape (144, 400, 400, 1), resize back to original shape

    return test_pred
