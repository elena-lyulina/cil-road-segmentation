import os
import cv2
from glob import glob
import numpy as np
import torch

from src.data.dataset import load_all_from_path, np_to_tensor
from src.models.small_UNet.small_UNet import UNet

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = './checkpoints/model_xxx.pth'
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path))

    ROOT_PATH = "../../"
    test_path = os.path.join(ROOT_PATH, 'data', 'test', 'images')
    test_paths = glob(test_path + '/*.png')

    test_images = load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]
    test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in test_images], 0)
    test_images = test_images[:, :, :, :3]
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)