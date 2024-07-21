import os
import shutil

import cv2
import numpy as np

if __name__ == '__main__':
    img_folder = '/data/CIL/90k/images'
    mask_folder = '/data/CIL/90k/masks'

    out_image_folder = '/data/CIL/90k/filtered/images'
    out_masks_folder = '/data/CIL/90k/filtered/masks'

    img_directory = os.fsencode(img_folder)
    mask_directory = os.fsencode(mask_folder)

    for file in os.listdir(img_directory):
        mask_path = os.path.join(mask_folder, os.fsdecode(file))
        image_path = os.path.join(img_folder, os.fsdecode(file))

        mask = cv2.imread(mask_path)
        average = np.average(np.array(mask, dtype=float), axis=(0, 1, 2))

        if average > 5:
            shutil.copy(mask_path, out_masks_folder)
            shutil.copy(image_path, out_image_folder)
