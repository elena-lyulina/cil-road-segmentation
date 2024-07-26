import json
import os
import shutil

if __name__ == '__main__':
    cil_dict = json.load(open('M:\\Workspace\\cil-road-segmentation\\data\\cil\\CLIP_clusters.json'))

    img_folder = 'M:\\CIL_datasets\\CIL\\training\\images'
    mask_folder = 'M:\\CIL_datasets\\CIL\\training\\groundtruth'

    img_folder_cluster0 = 'M:\\CIL_datasets\\CIL\\cluster0\\images'
    mask_folder_cluster0 = 'M:\\CIL_datasets\\CIL\\cluster0\\masks'

    img_folder_cluster1 = 'M:\\CIL_datasets\\CIL\\cluster1\\images'
    mask_folder_cluster1 = 'M:\\CIL_datasets\\CIL\\cluster1\\masks'

    img_directory = os.fsencode(img_folder)
    for file in os.listdir(img_directory):
        mask_path = os.path.join(mask_folder, os.fsdecode(file))
        image_path = os.path.join(img_folder, os.fsdecode(file))

        name = 'cil/training/images/' + os.fsdecode(file)
        cluster = cil_dict[name]

        if cluster == 0:
            shutil.copy(image_path, img_folder_cluster0)
            shutil.copy(mask_path, mask_folder_cluster0)
        elif cluster == 1:
            shutil.copy(image_path, img_folder_cluster1)
            shutil.copy(mask_path, mask_folder_cluster1)
        else:
            raise Exception('Why?')
