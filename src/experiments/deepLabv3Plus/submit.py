from src.constants import SRC_PATH

from src.submission.test import test_on_full_images

if __name__ == '__main__':
    config_path = SRC_PATH.joinpath('/ws/cil_checkpoints/deepLab/cil/deeplabv3plus_cil_pretrained_all_both_clusters.json')
    test_on_full_images(config_path)