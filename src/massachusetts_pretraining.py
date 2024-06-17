import os
import numpy as np
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from models_import import DC_Unet, UNet
from torch.utils.data import Dataset

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
class Massachusetts(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(glob(os.path.join(image_dir, '*.tiff')))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.tif')))

        self.crop_size = 384
        self.image_size = 1500
        self.indices = []
        

        ### Allows 9 patches 384x384 for each 1500x1500 original to be used, only compute once the double loop
        for i in range(len(self.image_paths)):
            for y in range(0, self.image_size, self.crop_size):
                for x in range(0, self.image_size, self.crop_size):
                    if y + self.crop_size <= self.image_size and x + self.crop_size <= self.image_size:
                        self.indices.append((i, x, y))
    
    def __len__(self):
        return len(self.indices)  ### allows all crops to be used without change to dataloading or batch handling
    
    def __getitem__(self, idx):
        image_idx, x, y = self.indices[idx]
        img_path = self.image_paths[image_idx]
        mask_path = self.mask_paths[image_idx]
        
        image = np.array(Image.open(img_path), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.float32)
        

        image /= 255.0
        mask /= 255.0
        
        # crop to 384x384 as used in main dataset
        image_crop = image[y:y+self.crop_size, x:x+self.crop_size, :]
        mask_crop = mask[y:y+self.crop_size, x:x+self.crop_size]
        
        #CHW for torch
        image_crop = np.transpose(image_crop, (2, 0, 1))
        
        return torch.from_numpy(image_crop), torch.from_numpy(mask_crop)
    
train_dataset = Massachusetts(train_images_path, train_masks_path)
val_dataset = Massachusetts(test_images_path, test_masks_path)


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
