from pathlib import Path

from src.submission.end2end import end2end

if __name__ == '__main__':
    config_paths_with_clusters =[
        (Path('<model1_cluster0_path>'),
         Path('<model1_cluster01_path>')),
        (Path('<model2_cluster0_path>'),
         Path('<model2_cluster01_path>')),
        ...
    ]

    config_paths_without_clusters = [
        Path('<model1_path>'),
        Path('<model2_path>'),
        ...
    ]

    experiment_name = 'example'
    voter = 'hard_pixel'
    cluster = True
    with_mae = False
    mae_path = ''

    end2end(config_paths_with_clusters, voter, experiment_name=experiment_name, with_mae=with_mae, cluster=cluster, mae_config_path=mae_path)