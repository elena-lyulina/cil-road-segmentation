import torch
from torch import nn as nn


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2.0 * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=1.5, smooth=1e-4, batch=True):
        """
        Initializes the Focal Tversky Loss function.
        Original implementation at: https://github.com/nabsabraham/focal-tversky-unet
        Parameters:
            alpha (float): Weight of false negatives.
            gamma (float): Focusing parameter.
            smooth (float): Smoothing constant to avoid division by zero.
            batch (bool): Whether to compute the loss over the whole batch or per sample.
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.batch = batch

    def tversky_index(self, y_pred, y_true):
        """
        Compute the Tversky index.

        Parameters:
            y_pred (torch.Tensor): Predicted probabilities.
            y_true (torch.Tensor): Ground truth binary labels.

        Returns:
            torch.Tensor: Computed Tversky index.
        """
        if self.batch:
            true_pos = torch.sum(y_true * y_pred)
            false_neg = torch.sum(y_true * (1 - y_pred))
            false_pos = torch.sum((1 - y_true) * y_pred)
        else:
            true_pos = torch.sum(y_true * y_pred, dim=(1, 2, 3))
            false_neg = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
            false_pos = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))

        tversky = (true_pos + self.smooth) / (
            true_pos
            + self.alpha * false_neg
            + (1 - self.alpha) * false_pos
            + self.smooth
        )
        return tversky

    def __call__(self, y_pred, y_true):
        """
        Compute the Focal Tversky Loss.

        Parameters:
            y_pred (torch.Tensor): Predicted probabilities (after sigmoid/softmax).
            y_true (torch.Tensor): Ground truth binary labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        tversky = self.tversky_index(y_pred, y_true)
        focal_tversky_loss = torch.pow((1 - tversky), self.gamma)
        return focal_tversky_loss.mean()
