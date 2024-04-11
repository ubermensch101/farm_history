import numpy as np
import torch
import matplotlib.pyplot as plt


def show_images_with_labels(images, class_names):
        num_images = len(images)
        rows = int(np.sqrt(num_images))
        cols = int(np.ceil(num_images / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            if i < num_images:
                img = images[i].permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
                ax.imshow(img)
                ax.set_title(f"Class: {class_names[i]}")
                ax.axis('off')
            else:
                ax.axis('off')
        plt.tight_layout()
        plt.show()

def visualize(dataloader, class_names):
    images, labels = next(iter(dataloader))
    class_labels = [class_names[label] for label in labels]
    show_images_with_labels(images, class_labels)


def normalization_params(train_loader):
    mean_sum = np.zeros(3)
    std_sum = np.zeros(3)

    count = 0
    # Iterate through the training dataset to calculate the sum of pixel values
    for images, _ in train_loader:
        IMAGE_SIZE = images.size(2)*images.size(3)
        batch_size = images.size(0)
        mean_sum += np.sum(images.numpy(), axis=(0, 2, 3)) /IMAGE_SIZE
        std_sum += np.sum(images.numpy()**2, axis=(0, 2, 3)) /IMAGE_SIZE
        count += batch_size
        break

    # Calculate the mean and standard deviation
    mean = mean_sum / count
    std = np.sqrt((std_sum / count) - (mean ** 2))

    return mean, std


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def activate(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



class ToHistogram:
    def __init__(self, bins=256, min=0.0, max=1.0):
        self.bins = bins
        self.min = min
        self.max = max
    
    def __call__(self, img):

        histograms = torch.stack([torch.histc(channel, bins=self.bins, min=self.min, max=self.max) for channel in img])
        # Normalize histograms
        histograms = histograms / histograms.sum(dim=1, keepdim=True)
        return histograms


class ToHSV:
    def __init__(self, ):
            pass
    
    def __call__(self, img):
        img = img.convert('HSV')
        return img





