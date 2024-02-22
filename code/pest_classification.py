import os
import pandas as pd
# import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from types import SimpleNamespace
# config = SimpleNamespace(**{})


# CCMT path
path = os.path.expanduser("~/data/ccmt/CCMT Dataset-Augmented")

# Collect file paths in a dataframe
crops = ["Maize"]  # just maize for now

crop_classes = {}
data = []

for crop in crops:
    # Loop through crop-specific classes
    crop_classes[crop] = os.listdir(os.path.join(path, crop, "train_set"))
    
    if ".DS_Store" in crop_classes[crop]:
        crop_classes[crop].remove(".DS_Store")

    for crop_class in crop_classes[crop]:
        # Loop through images in each class
        for set in ["train_set", "test_set"]:
            for roots, dirs, files in os.walk(os.path.join(path, crop, set, crop_class)):
                for file in files:
                    data.append((crop, set, crop_class, os.path.join(crop, set, crop_class, file)))
        
df = pd.DataFrame(data, columns=["crop", "set", "crop_class", "file"])  # convert to pandas


# Define dataset class
class AugmentedCCMT(Dataset):
    def __init__(self, config, df, transform=None, mode="val"):
        # Set attributes
        self.image_dir = config.image_dir
        self.df = df
        self.files = df["file"].values
        self.labels = df["crop_class"].values
        
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
