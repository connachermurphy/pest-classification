import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
import pandas as pd
# import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# from types import SimpleNamespace
# config = SimpleNamespace(**{})


# CCMT path
path = os.path.expanduser("~/data/ccmt/CCMT Dataset-Augmented")

# Collect file paths in a dataframe
crops = ["Maize"]  # just maize for now

crop_descriptions = {}
data = []

for crop in crops:
    # Loop through crop-specific classes
    crop_descriptions[crop] = os.listdir(os.path.join(path, crop, "train_set"))
    
    if ".DS_Store" in crop_descriptions[crop]:
        crop_descriptions[crop].remove(".DS_Store")

    for crop_class in crop_descriptions[crop]:
        # Loop through images in each class
        for set in ["train_set", "test_set"]:
            for roots, dirs, files in os.walk(os.path.join(path, crop, set, crop_class)):
                for file in files:
                    data.append((crop, set, crop_class, os.path.join(crop, set, crop_class, file)))
        
df = pd.DataFrame(data, columns=["crop", "set", "crop_description", "file"])  # convert to pandas
df["label"] = df["crop_description"].apply(lambda x: 1 if x == "Healthy" else 0)

# Define dataset class
class AugmentedCCMT(Dataset):
    def __init__(self, config, df, transform=None, mode="val"):
        # Set attributes
        self.image_dir = config.image_dir
        self.df = df
        self.files = df["file"].values
        self.labels = df["label"].values
        
        # Define transformation
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose(
                [
                    A.Resize(config.image_size, config.image_size),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get label and file_path for index
        label = self.labels[idx]
        file_path = os.path.join(self.image_dir, self.files[idx])

        # Read image at file_path with OpenCV
        image = cv2.imread(file_path)

        # Convert to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        augmented = self.transform(image=image)
        image = augmented["image"]

        # Normalize to unit interval
        image = image / 255

        return image, label
    

# Training step of a single epoch
def train_epoch(dataloader, model, optimizer, config):
    # Training mode
    model.train()

    epoch_loss_long = []

    for i, (images, labels) in tqdm(enumerate(dataloader), total = len(dataloader)):
        images = images.to(config.device)
        labels = labels.to(config.device)

        # Zero gradient
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = config.criterion(outputs, labels)

        # Backpropogation
        loss.backward()

        # Update weights
        optimizer.step()

        # Update loss
        epoch_loss_long.append(loss.item())

    epoch_loss = np.mean(epoch_loss_long)
    
    return epoch_loss