from pathlib import Path

from src.submission.end2end import end2end

if __name__ == '__main__':
    # config_paths = [Path('/ws/cil_checkpoints/deepLab/hrnet/cil/deeplabv3plus_cil_pretrained_all_both_clusters_acc0-95_date28-07-2024_13-39-10_1.json'),
    #                 Path('/ws/cil_checkpoints/dinoPlusUNet/cil/dino_plus_unet_cil_pretrained_all_both_clusters_acc0-94_date29-07-2024_09-12-16_0.json'),
    #                 Path('/ws/cil_checkpoints/dLinkNet/cil/dLinkNet_cil_pretrained_all_both_clusters_acc0-94_date27-07-2024_17-18-04_8.json'),
    #                 Path('/ws/cil_checkpoints/PSPNet/cil/PSPNet_cil_pretrained_all_both_clusters_acc0-94_date27-07-2024_16-58-18_1.json'),
    #                 Path('/ws/cil_checkpoints/UNetPlusPlus/cil/unetplusplus_cil_pretrained_all_both_clusters_acc0-95_date28-07-2024_13-25-11_6.json')],

    # config_paths = [(Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\deepLab\\hrnet\\cil_cluster0\\deeplabv3plus_cil_cluster0_pretrained_all_cluster0_acc0-95_date29-07-2024_22-45-26_2.json'),
    #                 Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\deepLab\\hrnet\\cil_cluster1\\deeplabv3plus_cil_cluster1_pretrained_all_cluster1_acc0-94_date28-07-2024_11-43-49_4.json'))]

    # config_paths = [(Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\deepLab\\resnet\\cil_cluster0\\deeplabv3plus_cil_cluster0_pretrained_all_cluster0_resnet_acc0-95_date28-07-2024_14-19-54_7.json'),
    #                  Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\deepLab\\resnet\\cil_cluster1\\deeplabv3plus_cil_cluster1_pretrained_all_cluster1_resnet_acc0-94_date28-07-2024_14-38-17_8.json'))]

    # config_paths = [(Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\UNetPlusPlus\\cil_cluster0\\unetplusplus_cil_cluster0_pretrained_all_cluster0_acc0-95_date28-07-2024_11-05-53_9.json'),
    #                  Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\UNetPlusPlus\\cil_cluster1\\unetplusplus_cil_cluster1_pretrained_all_cluster1_acc0-94_date28-07-2024_11-15-39_3.json'))]

    # config_paths = [(Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\dLinkNet\\cil_cluster0\\dLinkNet_cil_cluster0_pretrained_all_cluster0_acc0-95_date27-07-2024_17-30-15_1.json'),
    #                  Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\dLinkNet\\cil_cluster1\\dLinkNet_cil_cluster1_pretrained_all_cluster1_acc0-94_date27-07-2024_17-35-19_9.json'))]

    # config_paths = [(Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\PSPNet\\cil_cluster0\\PSPNet_cil_cluster0_pretrained_all_cluster0_acc0-95_date27-07-2024_17-02-00_2.json'),
    #                  Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\PSPNet\\cil_cluster1\\PSPNet_cil_cluster1_pretrained_all_cluster1_acc0-94_date27-07-2024_17-04-01_9.json'))]

    # config_paths = [(Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\dinoPlusUNet\\cil_cluster0\\dino_plus_unet_cil_cluster0_pretrained_all_cluster0_acc0-95_date29-07-2024_10-16-46_9.json'),
    #                  Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\dinoPlusUNet\\cil_cluster1\\dino_plus_unet_cil_cluster1_pretrained_all_cluster1_acc0-94_date29-07-2024_11-01-10_6.json'))]

    config_paths =[
        (Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\deepLab\\hrnet\\cil\\deeplabv3plus_cil_pretrained_all_both_clusters_acc0-95_date28-07-2024_13-39-10_1.json'),
         Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\deepLab\\hrnet\\cil\\deeplabv3plus_cil_pretrained_all_both_clusters_acc0-95_date28-07-2024_13-39-10_1.json')),

        (Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\deepLab\\resnet\\cil_cluster0\\deeplabv3plus_cil_cluster0_pretrained_all_cluster0_resnet_acc0-95_date28-07-2024_14-19-54_7.json'),
         Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\deepLab\\resnet\\cil_cluster1\\deeplabv3plus_cil_cluster1_pretrained_all_cluster1_resnet_acc0-94_date28-07-2024_14-38-17_8.json')),

        (Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\UNetPlusPlus\\cil_cluster0\\unetplusplus_cil_cluster0_pretrained_all_cluster0_acc0-95_date28-07-2024_11-05-53_9.json'),
         Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\UNetPlusPlus\\cil_cluster1\\unetplusplus_cil_cluster1_pretrained_all_cluster1_acc0-94_date28-07-2024_11-15-39_3.json')),

        (Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\dLinkNet\\cil_cluster0\\dLinkNet_cil_cluster0_pretrained_all_cluster0_acc0-95_date27-07-2024_17-30-15_1.json'),
         Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\dLinkNet\\cil_cluster1\\dLinkNet_cil_cluster1_pretrained_all_cluster1_acc0-94_date27-07-2024_17-35-19_9.json')),

        (Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\PSPNet\\cil_cluster0\\PSPNet_cil_cluster0_pretrained_all_cluster0_acc0-95_date27-07-2024_17-02-00_2.json'),
         Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\PSPNet\\cil_cluster1\\PSPNet_cil_cluster1_pretrained_all_cluster1_acc0-94_date27-07-2024_17-04-01_9.json')),

        (Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\dinoPlusUNet\\cil\\dino_plus_unet_cil_pretrained_all_both_clusters_acc0-94_date29-07-2024_09-12-16_0.json'),
         Path('C:\\Users\\Louis\\Desktop\\CIL_results_files-NEW\\dinoPlusUNet\\cil\\dino_plus_unet_cil_pretrained_all_both_clusters_acc0-94_date29-07-2024_09-12-16_0.json'))
    ]

    voter = 'hard_pixel'
    end2end(config_paths, voter, experiment_name='', with_mae=False, cluster=True)