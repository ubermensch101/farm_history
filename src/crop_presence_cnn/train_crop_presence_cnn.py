import os
import sys
import json
import torch
import subprocess
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms, datasets

from torchsummary import summary
import matplotlib.pyplot as plt
from src.crop_presence_cnn.train_utils import EarlyStopper,visualize
from src.crop_presence_cnn.models import *
## Reproducibility
torch.manual_seed(0)

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHECKPOINT_PATH = os.path.join(ROOT_DIR, "src", "crop_presence_cnn", "weights", "crop_presence_cnn.pt")

## CONFIG
dir_path = os.path.dirname(__file__)
setup_file = os.path.join(dir_path,"train_config.json")
with open(setup_file,'r') as file:
    config = json.loads(file.read())



if __name__ == "__main__":
    IMAGE_SIZE = config['img_size']

    mean = [0.38428715, 0.30033973, 0.18253809]
    std = [0.12181732, 0.07587652, 0.0720039 ]
    # create image loading transformations
    transform_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]),

        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) 
        ])

    transform_val_test = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) 
    ])

    # Load data from folders
    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform_train)
    # val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "val"), transform=transform_val_test)
    test_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=transform_val_test)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
    
    # # Visualize some data
    # visualize(train_loader, class_names)
    
    class_names = train_dataset.classes    

    model = SimpleCNN()
    # # Print model summary
    # summary(model, (3,64,64))

    # Train Model
    early_stopper = EarlyStopper(patience=7, min_delta=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    history = train_classifier(model, 
                        num_epochs=config["epochs"], 
                        criterion=criterion,
                        optimizer=optimizer, 
                        callbacks=[early_stopper],
                        train_loader=train_loader, 
                        val_loader=test_loader, 
                        device=config['device'])


    model.save(CHECKPOINT_PATH) 