import os
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from types import SimpleNamespace

class AugmentedCCMT(Dataset):
    def __init__(self, config, df, transform=None, mode="val"):
        self.image_dir = config.image_dir
        self.df = df
        self.files = df["file"].values
        self.labels = df["crop_class"].values

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
        # Get file_path and label for index
        label = self.labels[idx]
        file_path = os.path.join(self.image_dir, self.files[idx])

        # Read an image with OpenCV
        image = cv2.imread(file_path)

        # Convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentations
        augmented = self.transform(image=image)
        image = augmented["image"]

        # Normalize because ToTensorV2() doesn't normalize the image
        image = image / 255

        return image, label
