from src.models.utils import MODEL_REGISTRY
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import torch.optim as optim




@MODEL_REGISTRY.register('fcn_8')
class FCN8s(nn.Module):
    """Using pre-trained VGG for base model
    Source model: https://arxiv.org/pdf/1411.4038"""
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()

        # Load the pretrained VGG16 model
        vgg = models.vgg16(pretrained=True)

        # Encoder: VGG16 layers up to pool5
        self.features = vgg.features

        # Fully convolutional layers
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # Transposed convolutions for upsampling
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4)
        self.upscore16 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=32, stride=16, bias=False)

        # Scoring layers for skip connections
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # Forward pass through VGG16 feature layers
        pool3 = self.features[:17](x)  # up to relu4_3
        pool4 = self.features[17:24](pool3)  # up to relu5_3
        pool5 = self.features[24:](pool4)  # up to maxpool5
        # Stop here for FCN32s

        # Fully convolutional layers
        fc6 = self.fc6(pool5)
        fc6 = self.relu6(fc6)
        fc6 = self.drop6(fc6)

        fc7 = self.fc7(fc6)
        fc7 = self.relu7(fc7)
        fc7 = self.drop7(fc7)

        score_fr = self.score_fr(fc7)  # Output fo final scoring layer

        # Has twice the spatial resolution of score_fr
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(
            pool4)[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        upscore_pool4 = (upscore2 + score_pool4)  # Check if necessary
        # Stop here for FCN16s

        upscore_pool16 = self.upscore16(upscore_pool4)
        score_pool3 = self.score_pool3(
            pool3)[:, :, 9:9 + upscore_pool16.size()[2], 9:9 + upscore_pool16.size()[3]]
        upscore8 = self.upscore8(upscore_pool16 + score_pool3)
        out = upscore8[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]]

        return out


def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs):
    # Initialize lists to keep track of metrics
    train_loss_history = []
    val_loss_history = []
    train_metrics_history = {metric: [] for metric in metric_fns}
    val_metrics_history = {metric: [] for metric in metric_fns}

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_metrics = {metric: 0.0 for metric in metric_fns}
        
        for i, (inputs, targets) in enumerate(train_dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Log partial metrics
            running_loss += loss.item()
            for metric_name, metric_fn in metric_fns.items():
                running_metrics[metric_name] += metric_fn(outputs, targets).item()


        # Collect metrics at the end of the epoch
        train_loss_history.append(running_loss / len(train_dataloader))
        for metric_name in metric_fns:
            train_metrics_history[metric_name].append(running_metrics[metric_name] / len(train_dataloader))

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_metrics = {metric: 0.0 for metric in metric_fns}
        
        with torch.no_grad():
            for inputs, targets in eval_dataloader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                val_running_loss += loss.item()
                for metric_name, metric_fn in metric_fns.items():
                    val_running_metrics[metric_name] += metric_fn(outputs, targets).item()

        # Log partial metrics for validation
        val_loss_history.append(val_running_loss / len(eval_dataloader))
        for metric_name in metric_fns:
            val_metrics_history[metric_name].append(val_running_metrics[metric_name] / len(eval_dataloader))

        # Log to TensorBoard
        
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss_history[-1]}, Validation Loss: {val_loss_history[-1]}')

    print('Finished Training, saving model.')
    model_path = './models/fcn_8/checkpoints'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_path + f'/model_{time.strftime("%Y%m%d-%H%M%S")}.pth')
    print('Model saved.')

#     return model, train_loss_history, val_loss_history, train_metrics_history, val_metrics_history


# def test():
#     model = FCN8s(num_classes=21)
#     # Batch size 1, 3 color channels, 224x224 image
#     input_tensor = torch.randn(1, 3, 224, 224)
#     output = model(input_tensor)
#     print(output.shape) 

#     # test-train
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     metric_fns = {
#         'accuracy': lambda outputs, targets: (outputs.argmax(1) == targets).float().mean()
#     }
#     train_dataloader = ...  # Replace with actual DataLoader
#     eval_dataloader = ...   # Replace with actual DataLoader

#     n_epochs = 10
#     train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)
