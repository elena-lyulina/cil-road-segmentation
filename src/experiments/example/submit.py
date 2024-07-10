from src.constants import SRC_PATH

from src.submission.test import test_on_full_images

if __name__ == '__main__':
    config_path = SRC_PATH.joinpath('experiments/example/results/small_unet_cil_from_CIL_notebook_acc0-76_date06-07-2024_21-50-48_0.json')
    test_on_full_images(config_path)