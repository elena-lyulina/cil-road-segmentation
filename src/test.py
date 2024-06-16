import os
import cv2
from glob import glob
from PIL import Image
import numpy as np
import torch

from src.data.dataset import load_all_from_path, np_to_tensor
from src.models.small_UNet.small_UNet import UNet
from src.submission.mask_to_submission import masks_to_submission

if __name__ == '__main__':
    PATCH_SIZE = 16
    CUTOFF = 0.25

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = './models/small_UNet/checkpoints/model_20240615-203721.pth'
    prediction_path = './../out/predictions'
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)
    model = UNet().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    ROOT_PATH = "../"
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

    for i, img in enumerate(test_pred):
        img_name = test_paths[i].split('/')[-1].split('\\')[-1]
        file_path = os.path.join(prediction_path, img_name)
        img = cv2.resize(img, dsize=size)
        image = Image.fromarray((img*255).astype(np.uint8))
        image.save(file_path)

    masks_to_submission('./../out/small_unet_submission.csv', None, glob(prediction_path + '/*png'))