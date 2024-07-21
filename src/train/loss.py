"""
Most loss functions are taken from https://www.sciencedirect.com/science/article/pii/S1569843222003478
"""

import torch
from torch import nn as nn
from skimage.morphology import skeletonize


class SoftDiceLoss(nn.Module):
    def __init__(self, batch=True, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.batch = batch
        self.smooth = smooth

    def dice_coeff(self, y_pred, y_true):
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2.0 * intersection + self.smooth) / (i + j + self.smooth)
        return score.mean()

    def __call__(self, y_pred, y_true):
        loss = 1 - self.dice_coeff(y_pred, y_true)
        return loss


class SquaredDiceLoss(nn.Module):
    def __init__(self, batch=True, smooth=1.0):
        super(SquaredDiceLoss, self).__init__()
        self.batch = batch
        self.smooth = smooth

    def squared_dice_coeff(self, y_pred, y_true):
        if self.batch:
            i = torch.sum(torch.square(y_true))
            j = torch.sum(torch.square(y_pred))
            intersection = torch.sum(y_true * y_pred)
        else:
            i = torch.square(y_true).sum(1).sum(1).sum(1)
            j = torch.square(y_pred).sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2.0 * intersection + self.smooth) / (i + j + self.smooth)
        return score.mean()

    def __call__(self, y_pred, y_true):
        loss = 1 - self.squared_dice_coeff(y_pred, y_true)
        return loss


class LogCoshDiceLoss(nn.Module):
    def __init__(self, batch=True, smooth=1.0):
        super(LogCoshDiceLoss, self).__init__()
        self.batch = batch
        self.smooth = smooth
        self.dice_loss = SoftDiceLoss(batch, self.smooth)

    def __call__(self, y_pred, y_true):
        dice = self.dice_loss(y_pred, y_true)
        return torch.log(torch.cosh(dice))


class CenterlineDiceLoss(nn.Module):
    """
    Centerline Dice Loss. First introduced in: https://arxiv.org/pdf/2003.07311
    """

    def __init__(self, iter_=10, smooth=1.0):
        super(CenterlineDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def binarize(self, tensor):
        return (tensor > 0.5).float()

    def __call__(self, y_pred, y_true, skel_true=None):
        y_pred_bin = self.binarize(y_pred).cpu().numpy()
        y_true_bin = self.binarize(y_true).cpu().numpy()

        skel_pred = (
            torch.tensor([skeletonize(y) for y in y_pred_bin]).float().to(y_pred.device)
        )
        if skel_true is None:
            skel_true = (
                torch.tensor([skeletonize(y) for y in y_true_bin])
                .float()
                .to(y_true.device)
            )
        else:
            skel_true = self.binarize(skel_true).cpu().numpy()
            skel_true = (
                torch.tensor([skeletonize(y) for y in skel_true])
                .float()
                .to(y_true.device)
            )

        tprec = (
            torch.sum(torch.multiply(skel_pred, y_true)[:, 1:, ...]) + self.smooth
        ) / (torch.sum(skel_pred[:, 1:, ...]) + self.smooth)
        tsens = (
            torch.sum(torch.multiply(skel_true, y_pred)[:, 1:, ...]) + self.smooth
        ) / (torch.sum(skel_true[:, 1:, ...]) + self.smooth)
        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice.mean()


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
        tversky = self.tversky_index(y_pred, y_true)
        focal_tversky_loss = torch.pow((1 - tversky), self.gamma)
        return focal_tversky_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, loss1, loss2, llambda=1.0):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.llambda = llambda

    def __call__(self, y_pred, y_true):
        loss1_value = self.loss1(y_pred, y_true)
        loss2_value = self.loss2(y_pred, y_true)
        return loss1_value + self.llambda * loss2_value
