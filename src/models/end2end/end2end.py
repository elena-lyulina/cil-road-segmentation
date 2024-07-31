import numpy as np
from torch import nn
import json
import torch
from typing import List
from pathlib import Path
from src import voter
import numpy as np

from src.models.utils import MODEL_REGISTRY
from src.train.train import load_checkpoint


@MODEL_REGISTRY.register("end2end")
class End2End(nn.Module):

    def __init__(self, config_paths: List[Path], voter: str, train_mae: bool, mode: str,  resize_to=(400, 400)):
        super(End2End, self).__init__()

        self.sota_models_cluster0, self.sota_models_cluster1 = load_sota_models(config_paths[:-1], train_mae, resize_to)
        self.voter = voter
        self.mae = load_MAE(config_paths[-1], train_mae)
        self.mode = mode
    
    def forward(self, x):
        ensemble_predictions = self.ensemble_forward(x)

        voter_result = None

        if 'voter-then-mae' in self.mode:
            voter_result = self.vote(ensemble_predictions).unsqueeze(1)
            y_hat = self.mae(voter_result)
            if 'debug' in self.mode:
                if type(y_hat) == tuple:
                    y_hat = y_hat[0]
                y_hat = y_hat, voter_result
        
        elif self.mode == 'mae-then-voter':
            mae_all_predictions = [self.mae(prediction) for prediction in ensemble_predictions]
            y_hat = self.vote(mae_all_predictions)

        elif self.mode == 'no-mae':
            y_hat =  self.vote(ensemble_predictions).unsqueeze(1)
        
        else:
            raise ValueError("Invalid mode. Choose 'voter-then-mae', 'mae-then-voter', 'no-mae', 'voter-then-mae-debug'.")
        
        return y_hat

    def ensemble_forward(self, x):
        if isinstance(x, tuple):
            x, cluster_ids = x
            cluster_ids = cluster_ids.squeeze().tolist()
        else:
            raise ValueError("Expected a tuple (x, cluster_ids). Talk to Diego")

        x_list = list(torch.unbind(x, dim=0))
        ensemble_all_predictions = []

        for model0, model1 in zip(self.sota_models_cluster0, self.sota_models_cluster1):
            #model0 and model1 are the same architecture
            current_image_predictions = []
            for image, cluster_id in zip(x_list, cluster_ids):
                image = torch.stack((image, image), dim=0)
                if cluster_id == 0:
                    pred = model0(image)
                elif cluster_id == 1:
                    pred = model1(image)
                else:
                    raise ValueError("Invalid cluster id")
                pred = pred[0]
                current_image_predictions.append(pred)
            ensemble_all_predictions.append(torch.stack(current_image_predictions))
        return ensemble_all_predictions
            

    def vote(self, predictions):
        def reshape_prediction(prediction):
            return prediction.reshape(-1, 400, 400)

        try:
            return voter.__dict__[self.voter](list(map(reshape_prediction, predictions)))
        except KeyError:
            raise ValueError("Invalid voter type")
                

def load_sota_models(config_paths: List[Path], train_mae, resize_to):
    from src.experiments.config import load_config, get_model_path_from_config, load_checkpoint

    #check that every two consecutive models are the same architecture
    for i in range(0, len(config_paths), 2):
        config0 = load_config(Path(config_paths[i]))
        config1 = load_config(Path(config_paths[i + 1]))
        if config0['model']['name'] != config1['model']['name']:
            raise ValueError("Consecutive Models must be the same architecture")

    sota_models_cluster0 = []
    sota_models_cluster1 = []

    for config_path in config_paths:
        config_name = config_path
        config = load_config(Path(config_path))

        #TODO: is the following code really necessary?
        if config['dataset']['name'] == 'cil':
            real_resize_to = config['dataset']['params'].get('resize_to')
            if real_resize_to is not None:
                print(
                    f"Found the size of the CIL images the model was trained on: {real_resize_to}")
                if resize_to != real_resize_to:
                    print(
                        f"Using the found size {real_resize_to} instead of given {resize_to}")
                    resize_to = real_resize_to


        # load pretrained sota model
        model_path = get_model_path_from_config(Path(config_path))
        model, _ = load_checkpoint(model_path)
        if train_mae:
            model.train()
        else:
            model.eval()

        for param in model.parameters():
            param.requires_grad = False

        # add model to the corresponding cluster
        if 'luster0' in config_name:
            sota_models_cluster0.append(model)
        elif 'luster1' in config_name:
            sota_models_cluster1.append(model)
        else:
            raise ValueError("Model can't be assigned to a cluster. 'luster0' or 'luster1' not found in the config path")
        
    return sota_models_cluster0, sota_models_cluster1


def load_MAE(mae_config_path: Path, train_mae: bool):
    from src.experiments.config import get_model_path_from_config, load_checkpoint

    model_path = get_model_path_from_config(Path(mae_config_path))
    model, _ = load_checkpoint(model_path)

    if train_mae:
        model.train()
        for param in model.parameters():
            param.requires_grad = True
    else:
        model.eval()

    return model