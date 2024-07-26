from src.constants import SRC_PATH

from src.submission.test import test_on_full_images

if __name__ == '__main__':
    config_path = SRC_PATH.joinpath('')
    test_on_full_images(config_path)