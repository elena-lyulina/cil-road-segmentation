import numpy as np

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

    all_predictions_expanded = np.kron(
        all_predictions_patches, np.ones((16, 16), dtype=np.float32))

    return all_predictions_expanded

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

    all_predictions_expanded = np.kron(
        all_predictions_patches, np.ones((16, 16), dtype=np.float32))

    return all_predictions_expanded
