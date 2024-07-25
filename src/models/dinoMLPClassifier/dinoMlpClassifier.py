import json
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from src.constants import DATA_PATH, DEVICE, ROOT_PATH, EXPERIMENTS_PATH


def load_DINO_model():
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2_vits14.to(DEVICE)
    return dinov2_vits14


def load_images(paths):
    images = [np.array(Image.open(f).convert('RGB')) for f in paths]
    transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])
    images = np.stack([transform_image(img)[:3] for img in images])
    return torch.from_numpy(images)


def compute_DINO_embeddings(images_paths: list, save_folder_path=None, batch_size=64, file_name='DINO_embeddings'):
    # computes DINO embeddings on given images and dumps them into a file if save_folder_path is not None
    if save_folder_path:
        os.makedirs(os.path.join(save_folder_path, 'embeddings'), exist_ok=True)

    # images_paths = get_img_paths(load_folder_path)
    print(f'\n{len(images_paths)}')

    result = []
    model = load_DINO_model()

    with torch.no_grad():
        for i in tqdm(range(0, len(images_paths), batch_size)):
            images_paths_batch = images_paths[i:i + batch_size]
            size = len(images_paths_batch)
            images_batch = load_images(images_paths_batch).to(DEVICE)
            embeddings = model(images_batch)

            images_paths_batch = [path.replace(ROOT_PATH, '') for path in images_paths_batch]
            embeddings = np.array(embeddings.cpu().numpy())
            # print(embeddings.shape)
            result += list(zip(images_paths_batch, embeddings.tolist()))

    result_dict = dict(result)
    if save_folder_path:
        with open(os.path.join(save_folder_path, f'{file_name}.json'), "w") as f:
            f.write(json.dumps(result_dict))


def train_MLP_classifier(save_path=None):
    with open(os.path.join(DATA_PATH, 'cil', 'CLIP_clusters.json'), "r") as f:
        # 'CLIP_clusters.json' stores labels for every image, obtained by CLIP + kmeans (in the cil-data.ipynb jupyter notebook)
        labels = json.load(f)

    with open(os.path.join(DATA_PATH, 'cil', 'DINO_embeddings_all.json'), "r") as f:
        # 'DINO_embeddings_all.json' stores DINO embeddings for every image obtained by #compute_DINO_embeddings
        embeddings = json.load(f)

    data = [(embeddings[key], val) for key, val in labels.items()]
    X, y = zip(*data)
    X = np.array(X)
    X = X.reshape(X.shape[0], -1)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(256, 64),
                        max_iter=1000, random_state=42)

    # Train the model on the training data
    mlp.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = mlp.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # save
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(mlp, f)

    return mlp


def load_MLP_classifier(load_path=os.path.join(EXPERIMENTS_PATH, 'CIL_clusters', 'results', 'CIL_clusters_MLP_classifier.pkl')):
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def predict_clusters(image_paths, DINO_model=None, MLP_classifier=None):
    # returns an array of predicted clusters for every image in image_paths
    # to not load models everytime, you can load them in advance and pass already loaded versions
    if DINO_model is None:
        DINO_model = load_DINO_model()

    if MLP_classifier is None:
        MLP_classifier = load_MLP_classifier()

    with torch.no_grad():
        images = load_images(image_paths).to(DEVICE)
        embeddings = DINO_model(images)
        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

        # to predict probability of every cluster
        # MLP_classifier.predict_proba(embeddings)

        return MLP_classifier.predict(embeddings)
