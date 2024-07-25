import os

from src.constants import DATA_PATH
from src.models.dinoMLPClassifier.dinoMlpClassifier import predict_clusters

if __name__ == '__main__':
    # test predictions
    image_paths = [os.path.join(DATA_PATH, 'cil', 'test', 'images', name) for name in ['satimage_144.png', 'satimage_145.png']]
    # A trained version will be loaded from 'CIL_clusters_MLP_classifier.pkl' file
    clusters = predict_clusters(image_paths)
    print(clusters)
