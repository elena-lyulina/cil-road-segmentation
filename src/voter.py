import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from src.constants import DEVICE
from src.data.datahandler import get_datahandler_class
from src.models.utils import get_model_class
import json


# Load models and weights
def get_config_path(model_path):
    return next(model_path.glob("*.json"))


def get_weights_path(model_path):
    return next(model_path.glob("*.pth"))


def load_models_and_weights(model_paths):
    models = []
    for model_path in model_paths:
        config_path = get_config_path(model_path)
        weights_path = get_weights_path(model_path)

        # Load model config and instantiate model
        with open(config_path) as f:
            config = json.load(f)
        model_class = get_model_class(config["model_name"])
        model = model_class(config)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        models.append(model)
    return models


# Get dataloader
def get_dataloader(config):
    datahandler_class = get_datahandler_class(config["datahandler"])
    datahandler = datahandler_class(config["datahandler_config"])
    return datahandler.get_test_dataloader()


# Hard voting at pixel level
def hard_voting_pixel_level(models, dataloader):
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
        batch_preds = torch.zeros_like(labels)

        for model in models:
            preds = model(images)
            batch_preds += torch.sigmoid(preds) > 0.5

        batch_preds = (batch_preds >= (len(models) / 2)).float()
        all_predictions.append(batch_preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_predictions), np.concatenate(all_labels)


# Soft voting at pixel level
def soft_voting_pixel_level(models, dataloader):
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
        batch_preds = torch.zeros_like(labels, dtype=torch.float32)

        for model in models:
            preds = model(images)
            batch_preds += torch.sigmoid(preds)

        batch_preds = (batch_preds / len(models)) > 0.5
        all_predictions.append(batch_preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_predictions), np.concatenate(all_labels)


# Hard voting at 16x16 patch level
def hard_voting_patch_level(models, dataloader):
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
        batch_preds = torch.zeros_like(labels)

        for model in models:
            preds = model(images)
            batch_preds += torch.sigmoid(preds) > 0.5

        # Flatten the labels to the patch level
        labels = labels.view(labels.size(0), -1, 16, 16)
        labels = (labels.mean(dim=(2, 3)) > 0.25).float()

        # Sum votes for each 16x16 patch
        patch_preds = batch_preds.view(batch_preds.size(0), -1, 16, 16)
        patch_preds = patch_preds.sum(dim=(2, 3)) >= (0.25 * 256 * len(models))
        patch_preds = patch_preds.float()

        all_predictions.append(patch_preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_predictions), np.concatenate(all_labels)


# Soft voting at 16x16 patch level
def soft_voting_patch_level(models, dataloader):
    all_predictions = []
    all_labels = []

    for batch in dataloader:
        images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
        batch_preds = torch.zeros_like(labels, dtype=torch.float32)

        for model in models:
            preds = model(images)
            batch_preds += torch.sigmoid(preds)

        # Flatten the labels to the patch level
        labels = labels.view(labels.size(0), -1, 16, 16)
        labels = (labels.mean(dim=(2, 3)) > 0.25).float()

        # Average votes for each 16x16 patch
        patch_preds = batch_preds.view(batch_preds.size(0), -1, 16, 16)
        patch_preds = patch_preds.sum(dim=(2, 3)) >= (0.25 * 256 * len(models))
        patch_preds = patch_preds.float()

        all_predictions.append(patch_preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_predictions), np.concatenate(all_labels)


# Evaluate all four combinations and print results
def evaluate_voting(models, dataloader):
    results = {}

    print("Evaluating hard voting at pixel level...")
    preds, labels = hard_voting_pixel_level(models, dataloader)
    accuracy = accuracy_score(labels.flatten(), preds.flatten())
    f1 = f1_score(labels.flatten(), preds.flatten())
    results["hard_pixel"] = (accuracy, f1)
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    print("Evaluating soft voting at pixel level...")
    preds, labels = soft_voting_pixel_level(models, dataloader)
    accuracy = accuracy_score(labels.flatten(), preds.flatten())
    f1 = f1_score(labels.flatten(), preds.flatten())
    results["soft_pixel"] = (accuracy, f1)
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    print("Evaluating hard voting at patch level...")
    preds, labels = hard_voting_patch_level(models, dataloader)
    accuracy = accuracy_score(labels.flatten(), preds.flatten())
    f1 = f1_score(labels.flatten(), preds.flatten())
    results["hard_patch"] = (accuracy, f1)
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    print("Evaluating soft voting at patch level...")
    preds, labels = soft_voting_patch_level(models, dataloader)
    accuracy = accuracy_score(labels.flatten(), preds.flatten())
    f1 = f1_score(labels.flatten(), preds.flatten())
    results["soft_patch"] = (accuracy, f1)
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    best_combination = max(results, key=lambda x: results[x][1])
    print(
        f"Best combination is {best_combination} with Accuracy: {results[best_combination][0]:.4f} and F1: {results[best_combination][1]:.4f}"
    )


# Main function to execute the voting classifier
def main():
    experiment_root = Path("experiments")
    experiment_paths = [p for p in experiment_root.iterdir() if p.is_dir()]
    model_paths = [p / "results" for p in experiment_paths if (p / "results").is_dir()]

    models = load_models_and_weights(model_paths)

    # Assuming config is same for all models, taking from the first one
    with open("voter_config_for_data.json") as f:
        config = json.load(f)

    dataloader = get_dataloader(config)
    evaluate_voting(models, dataloader)


if __name__ == "__main__":
    main()
