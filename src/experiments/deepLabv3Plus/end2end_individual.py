from pathlib import Path

from src.submission.end2end import end2end

if __name__ == '__main__':
    config_paths = [Path('/ws/cil_checkpoints/deepLab/hrnet/cil_cluster0/deeplabv3plus_cil_cluster0_pretrained_all_cluster0_acc0-95_date28-07-2024_11-25-35_3.json'),
                    Path('/ws/cil_checkpoints/deepLab/hrnet/cil_cluster1/deeplabv3plus_cil_cluster1_pretrained_all_cluster1_acc0-94_date28-07-2024_11-43-49_4.json')]
    voter = 'hard_pixel'
    end2end(config_paths, voter, with_mae=False)