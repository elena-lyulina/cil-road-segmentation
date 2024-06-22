import os
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from models_import import UNet
from torch.utils.data import Dataset

from src.data.massachusetts_dataset import MassachusettsDataset

print("Current working directory:", os.getcwd())

#### ADAPT THESE ####
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" #set what nodes to use if relevant. Set to "0" if single GPU
parallelize = True #set to false if single GPU

BATCH_SIZE = 32
NUM_WORKERS = 16 # tune that to CPU capacities, keep high number of parallelization used
N_EPOCHS = 5

model = UNet() 
#### ADAPT THESE ####


#### DO NOT CHANGE ####
PATCH_SIZE = 16 
CUTOFF = 0.25
#### DO NOT CHANGE ####



#### PATHS ####
ROOT_PATH = "../data-massachusetts"#../../ext_train/"

train_images_path = ROOT_PATH+"train"
train_masks_path = ROOT_PATH+"train_labels"

test_images_path = ROOT_PATH+"test"
test_masks_path = ROOT_PATH+"test_labels"

OUT_MODELS_PATH = "./pretrained_models"
#### PATHS ####



################# DATA PREP ##############

train_dataset = MassachusettsDataset(train_images_path, train_masks_path)
val_dataset = MassachusettsDataset(test_images_path, test_masks_path)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = True, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = True, shuffle=True)
################# DATA PREP ##############




############# UTILITIES ##############
def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs, device):
    # training loop

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        # training
        model.train()
        for (x, y) in pbar:
            optimizer.zero_grad()  # zero out gradients
            x, y = x.to(device, non_blocking = True), y.to(device, non_blocking=True)
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y.unsqueeze(1))
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                x, y = x.to(device, non_blocking = True), y.to(device, non_blocking = True)
                y_hat = model(x)  # forward pass
                loss = loss_fn(y_hat, y.unsqueeze(1))

                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
            print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))

    print('Finished Training')

def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()

def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    #patches_hat = y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    patches_hat = torch.mean(y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE), (-1,-3)) > CUTOFF
    patches = y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF

    return (patches == patches_hat).float().mean()
############# UTILITIES ##############




############# MAIN BUT I'M STUPID ########

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if parallelize:
    model = nn.DataParallel(model)
model.to(device)

loss_fn = nn.BCELoss()
metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
optimizer = torch.optim.Adam(model.parameters())

train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, N_EPOCHS, device)

if parallelize:
    torch.save(model.module.state_dict(), 'OUT_MODELS_PATH')
else:
    torch.save(model.state_dict(), 'OUT_MODELS_PATH')

############# MAIN BUT I'M STUPID ########
