from pathlib import Path

from src.submission.end2end import end2end

if __name__ == '__main__':
    # config_paths = [Path('/ws/cil_checkpoints/deepLab/hrnet/cil/deeplabv3plus_cil_pretrained_all_both_clusters_acc0-95_date28-07-2024_13-39-10_1.json'),
    #                 Path('/ws/cil_checkpoints/dinoPlusUNet/cil/dino_plus_unet_cil_pretrained_all_both_clusters_acc0-94_date29-07-2024_09-12-16_0.json'),
    #                 Path('/ws/cil_checkpoints/dLinkNet/cil/dLinkNet_cil_pretrained_all_both_clusters_acc0-94_date27-07-2024_17-18-04_8.json'),
    #                 Path('/ws/cil_checkpoints/PSPNet/cil/PSPNet_cil_pretrained_all_both_clusters_acc0-94_date27-07-2024_16-58-18_1.json'),
    #                 Path('/ws/cil_checkpoints/UNetPlusPlus/cil/unetplusplus_cil_pretrained_all_both_clusters_acc0-95_date28-07-2024_13-25-11_6.json')],

    config_paths = [Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\deepLab\\hrnet\\cildeeplabv3plus_cil_pretrained_all_both_clusters_acc0-95_date28-07-2024_13-39-10_1.json')]

    voter = 'hard_pixel'
    end2end(config_paths, voter, experiment_name='test', with_mae=False, cluster=False)