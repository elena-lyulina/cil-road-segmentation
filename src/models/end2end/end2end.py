from torch import nn
import json
from typing import List
from pathlib import Path
from src.experiments.config import load_config, get_model_path_from_config, load_checkpoint
from src import voter

from src.models.utils import MODEL_REGISTRY

@MODEL_REGISTRY.register("end2end")
class End2End(nn.Module):

    def __init__(self, config_paths: List[Path], voter: str, train_mae: bool, mode: str,  resize_to=(400, 400)):
        super(End2End, self).__init__()

        self.sota_models_cluster0, self.sota_models_cluster1 = load_sota_models(config_paths[:-1], train_mae, resize_to)
        self.voter = voter
        self.mae = load_MAE(config_paths[-1], train_mae)
        self.mode = mode
    
    def forward(self, x):
        if isinstance(x, tuple):
            x, cluster_id = x
        else:
            cluster_id = None
        
        if self.mode == 'voter-then-mae':
            if cluster_id == 0:
                predictions = [model(x) for model in self.sota_models_cluster0]
            elif cluster_id == 1:
                predictions = [model(x) for model in self.sota_models_cluster1]
            else:
                raise ValueError("cluster_id not found. This dataset is not supported. Dataset.__getitem__ must return a tuple with the cluster id. Check cluster1.py for reference.")
            
            try:
                predictions = voter.__dict__[self.voter](predictions)
            except KeyError:
                raise ValueError("Invalid voter type")
            
            return self.mae(predictions)
        
        elif self.mode == 'mae-then-voter':
            if cluster_id == 0:
                predictions = [model(x) for model in self.sota_models_cluster0]
            elif cluster_id == 1:
                predictions = [model(x) for model in self.sota_models_cluster1]
            else:
                raise ValueError("cluster_id not found. This dataset is not supported. Dataset.__getitem__ must return a tuple with the cluster id. Check cluster1.py for reference.")
            
            predictions = [self.mae(prediction) for prediction in predictions]

            try:
                return voter.__dict__[self.voter](predictions)
            except KeyError:
                raise ValueError("Invalid voter type")
                

def load_sota_models(config_paths: List[Path], train_mae, resize_to):
    sota_models_cluster0 = []
    sota_models_cluster1 = []

    for config_path in config_paths:
        config = json.loads(config_path.read_bytes())

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
        model_path = get_model_path_from_config(config_path)
        model, _ = load_checkpoint(model_path)
        if train_mae:
            model.train()
        else:
            model.eval()

        for param in model.parameters():
            param.requires_grad = False

        # add model to the corresponding cluster
        if 'luster0' in config_path:
            sota_models_cluster0.append(model)
        elif 'luster1' in config_path:
            sota_models_cluster1.append(model)
        else:
            raise ValueError("Model can't be assigned to a cluster. 'luster0' or 'luster1' not found in the config path")
        
    return sota_models_cluster0, sota_models_cluster1


def load_MAE(mae_config_path: Path, train_mae: bool):
    model_path = get_model_path_from_config(mae_config_path)
    model, _ = load_checkpoint(model_path)

    if train_mae:
        model.train()
        for param in model.parameters():
            param.requires_grad = True
    else:
        model.eval()

    return model