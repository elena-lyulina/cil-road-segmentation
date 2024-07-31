import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import random

from skimage import exposure
from numpy.random import default_rng

from src.data.utils import np_to_tensor


class RoadSegDataset(Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(
        self,
        image_paths,
        mask_paths,
        cutoff,
        device,
        resize_to=(400, 400),
        augment=None,
        masking_params = {
            "num_zero_patches": 8,
            "zero_patch_size": 50,
            "num_flip_patches": 25,
            "flip_patch_size": 16,
            "noise_threshold": 0.5,
        }
    ):
        if augment is None:
            augment = []
        self.items = list(zip(image_paths, mask_paths))
        self.cutoff = cutoff
        self.device = device
        self.resize_to = resize_to
        self.augment = augment
        self.masking_params = masking_params

        self.geometric_transform = A.OneOf(
            [
                A.Rotate(limit=0, always_apply=True),  # 0 degrees
                A.Rotate(limit=90, always_apply=True),  # 90 degrees
                A.Rotate(limit=180, always_apply=True),  # 180 degrees
                A.Rotate(limit=270, always_apply=True),  # 270 degrees
                A.HorizontalFlip(always_apply=True),  # Horizontal reflection
                A.VerticalFlip(always_apply=True),  # Vertical reflection
                A.Transpose(always_apply=True),  # Diagonal reflection across main diagonal
                # Diagonal reflection across secondary diagonal (not directly supported, combine transpose and flip)
                A.Compose([A.Transpose(always_apply=True), A.VerticalFlip(always_apply=True)])
            ],
            p=1,
        )

        self.color_transform = A.Compose(
            [
                # params should be a range, we might want to look into it later to tune these params or use defaults
                A.ColorJitter(brightness=(0.2, 0.2), contrast=(0.2, 0.2)),
                A.AdvancedBlur(
                    blur_limit=(5, 9), sigma_x_limit=(0.1, 5), sigma_y_limit=(0.1, 5)
                ),
            ]
        )

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        if "geometric" in self.augment:
            z = self.geometric_transform(image=x, mask=y)
            x = z["image"]
            y = z["mask"]

        if "color" in self.augment:
            x = self.color_transform(image=x)["image"]

        if "masked" in self.augment:
            x = self.apply_masking(y.copy())

        # cv2.imwrite('x.png', x)
        # cv2.imwrite('y.png', y)

        # cv2.imshow('x', x)
        # cv2.imshow('y', y)
        # cv2.waitKey(0)
        return x, y

    def apply_masking(self, mask):
        #read masking params into local variables, check if they are present in masking_params
        return blobs_and_erode(mask)


    def __getitem__(self, item):
        # return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))
        img_path, mask_path = self.items[item]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.array(image)[:, :, :3].astype(np.float32) / 255.0
        mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0

        if self.resize_to != (image.shape[0], image.shape[1]):  # resize images
            image = cv2.resize(image, dsize=self.resize_to)
            mask = cv2.resize(mask, dsize=self.resize_to)

        image, mask = self._preprocess(image, mask)
        if "masked" in self.augment:
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        image = np.moveaxis(
            image, -1, 0
        )  # pytorch works with CHW format instead of HWC
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        mask = np.moveaxis(mask, -1, 0)  # pytorch works with CHW format instead of HWC

        image_tensor = np_to_tensor(image, 'cpu')
        mask_tensor = np_to_tensor(mask, 'cpu')

        return image_tensor, mask_tensor

    def __len__(self):
        return len(self.items)
    

def add_random_black_blobs(img, seed=31, blur_sigma=7, threshold=179, kernel_size=(9, 9)):
    import skimage.exposure

    # Load the image in grayscale
    height, width = img.shape

    # Set the random seed and create noise
    rng = default_rng(seed=seed)
    noise = rng.integers(0, 255, (height, width), np.uint8, True)

    # Apply Gaussian blur to the noise
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma, borderType=cv2.BORDER_DEFAULT)

    # Stretch the histogram to full range
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)

    # Apply a threshold to create binary blobs
    thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY)[1]

    # Morphological operations to clean up the blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Since it's a black and white image, invert the mask to get black blobs
    black_blobs = 255 - mask

    # Combine the blobs with the original image
    result = cv2.bitwise_and(img, black_blobs)

    return result


def uniformly_erode_and_smooth_blobs(image, num_blobs=10, max_blob_size=50, erosion_size=5, dilation_size=5, seed=None):
    """
    Uniformly erodes and then smooths white regions in a black and white mask in blob-shaped patches.

    Parameters:
    image (numpy array): The input black and white mask.
    num_blobs (int): Number of random blobs to erode and smooth uniformly.
    max_blob_size (int): Maximum size of the blob in pixels.
    erosion_size (int): The size of the erosion kernel.
    dilation_size (int): The size of the dilation kernel to smooth edges.
    seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
    numpy array: The uniformly eroded and smoothed mask.
    """
    # Add border to avoid edge artifacts
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,0])

    # Ensure the image is in binary format
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    height, width = binary_image.shape
    rng = default_rng(seed=seed)

    # Define the erosion and dilation kernels as circular
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))

    # Create a temporary image for sequential processing
    temp_image = binary_image.copy()

    for _ in range(num_blobs):
        # Randomly choose the center of the blob
        x_center = rng.integers(0, width)
        y_center = rng.integers(0, height)
        blob_width = rng.integers(10, max_blob_size)
        blob_height = rng.integers(10, max_blob_size)

        # Create an elliptical/oval blob within a temporary mask
        temp_mask = np.zeros_like(binary_image)
        cv2.ellipse(temp_mask, (x_center, y_center), (blob_width // 2, blob_height // 2), 
                    angle=0, startAngle=0, endAngle=360, color=(255,), thickness=-1)

        # Apply erosion only to the region defined by the blob mask
        eroded_area = cv2.erode(temp_image, erosion_kernel, iterations=1)
        temp_image = np.where(temp_mask == 255, eroded_area, temp_image)

    # After all blobs have been eroded, apply dilation to smooth the entire image
    smoothed_image = cv2.dilate(temp_image, dilation_kernel, iterations=1)

    # Remove the added border
    smoothed_image = smoothed_image[10:-10, 10:-10]

    return smoothed_image

def noise(mask, noise_threshold=0.5):
    if noise_threshold < 100:
        noise = np.random.normal(0, 1, mask.shape)
        random_ones = (noise > noise_threshold).astype(np.float32)

        noise = np.random.normal(0, 1, mask.shape)
        random_zeros = (noise > noise_threshold).astype(np.float32)
    
        mask = mask + random_ones - random_zeros
        mask = np.clip(mask, 0, 1)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask


def blobs_and_erode(img):
    img = img.astype(np.uint8)
    img = add_random_black_blobs(img)
    img = uniformly_erode_and_smooth_blobs(img, num_blobs=2000, max_blob_size=50, erosion_size=2, dilation_size=5, seed=43)
    img = noise(img)

    return img



# Example usage
# img = cv2.imread('data/cil/training/groundtruth/satimage_10.png', cv2.IMREAD_GRAYSCALE)
# output_image = blobs_and_erode(img)

# # Show the image
# cv2.imshow('noise + erode', output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
