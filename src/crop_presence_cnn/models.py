import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

## Reproducibility
torch.manual_seed(0)

## TRAINING CONFIG
dir_path = os.path.dirname(__file__)
setup_file = os.path.join(dir_path,"train_config.json")
with open(setup_file,'r') as file:
    config = json.loads(file.read())

BATCH_SIZE = config['batch_size']
LR = config["lr"]
EPOCHS = config["epochs"]
DEVICE = config['device']
lambda_l2 = config['lambda_l2']



class SimpleClassifier(nn.Module):
    def __init__(self, bins=20):
        super().__init__()  
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(3*bins, 8)
        self.fc2 = nn.Linear(8, 2)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

    def save(self, filepath):
        torch.save(self, filepath)

    def load(self, filepath):
        try:
            self.load_state_dict(torch.load(filepath))
            self.eval()
        except FileNotFoundError as e:
            print("Error Loading Model: ", e)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()  
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p=0.2)  
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(p=0.3)  
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(p=0.3)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout(p=0.4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.fc1 = nn.Linear(256, 32)  
        self.fc2 = nn.Linear(32,2)

    def forward(self, x):

        ## Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)  
        x = self.pool(x)

        ## Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x) 
        x = self.pool(x)

        ## Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x) 
        x = self.pool(x)

        ## Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout4(x)
    
        ## GAP and Classifier
        x = self.global_avg_pool(x)
        x = x.view(-1, 256) 
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def save(self, filepath):
        # Create a dictionary to store necessary information
        checkpoint = {
            'state_dict': self.state_dict(),
        }
    
        torch.save(checkpoint, filepath)


    def load(self, filepath):
        try:
            model = torch.load(filepath)
            model.eval()
            return model
            
        except FileNotFoundError as e:
            print("Error Loading Model: ", e)



# def train_one_epoch(model, epoch_id, optimizer, criterion, train_loader, val_loader=None,callbacks=None, device="cpu"):
#     model = model.to(device)
#     total_loss = 0
#     correct = 0
#     total = 0
#     model.train()
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)      
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         loss = criterion(outputs, labels)
#         # Compute L2 regularization 
#         l2_regularization = 0
#         for param in model.parameters():
#             l2_regularization += torch.norm(param, p=2)
#         loss += lambda_l2 * l2_regularization
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     t_accuracy = correct / total

#     ## Validation
#     if val_loader:
#         model.eval()
#         with torch.no_grad():
#             correct = 0
#             total = 0
#             total_val_loss = 0

#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 loss = criterion(outputs, labels)
#                 total_val_loss +=loss.item()
        
#         v_accuracy = correct / total
        
#     return total_loss, t_accuracy, total_val_loss, v_accuracy
#     # scheduler.step(total_val_loss)


def train_classifier(model, num_epochs, optimizer, criterion, train_loader, val_loader=None,callbacks=None, device="cpu"):
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode="min", factor=0.5, patience=10)
    model = model.to(device)
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        log = ""
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)        ## Softmax
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            # Compute L2 regularization 
            l2_regularization = 0
            for param in model.parameters():
                l2_regularization += torch.norm(param, p=2)
            loss += lambda_l2 * l2_regularization
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        t_accuracy = correct / total
        train_accuracy.append(t_accuracy)
        train_loss.append(total_loss)
        log+= f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.5f},  Train Accuracy: {t_accuracy * 100:.2f}%  '

        ## Validation
        if val_loader:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                total_val_loss = 0

                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    total_val_loss +=loss.item()
            val_loss.append(total_val_loss)
            v_accuracy = correct / total
            val_accuracy.append(v_accuracy)
            log+= f'Validation Loss: {total_val_loss:.5f},  Val Accuracy: {v_accuracy * 100:.2f}%  ' 
        # scheduler.step(total_val_loss)
        print(log)

        ## Activate Callbacks
        if callbacks:
            for callback in callbacks:
                callback.activate(total_val_loss)

    normalized_train_losses = np.array(train_loss)/np.max(train_loss)
    normalized_val_losses = np.array(val_loss)/np.max(val_loss)
    train_accuracy = np.array(train_accuracy)
    val_accuracy = np.array(val_accuracy)

    # Plotting normalized losses
    plt.plot(normalized_train_losses, label='Training')
    plt.plot(normalized_val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.ylim(-0.05, 1.1) 
    plt.legend()
    # plt.savefig('loss.png')
    plt.show()
    plt.pause(1)
    plt.close()


    # Plotting accuracies
    plt.plot(train_accuracy, label="Train")
    plt.plot(val_accuracy, label="Validation")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(-0.05, 1.1) 
    plt.legend()
    # plt.savefig('accuracy.png')
    plt.show()
    plt.pause(1)
    plt.close()
    
    return {
        "train_loss": normalized_train_losses,
        "validation_loss": normalized_val_losses,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy
    }



