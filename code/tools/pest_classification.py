# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2
import numpy as np
import os
# from memory_profiler import profile
import pandas as pd
from PIL import Image
import random
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
# from types import SimpleNamespace
# config = SimpleNamespace(**{})


# CCMT path
path = os.path.expanduser("~/data/ccmt_proc_240311")

# Collect file paths in a dataframe
crops = ["Maize"]  # just maize for now

data = []

for crop in crops:
    # Loop through crop-specific classes
    crop_descriptions_temp = os.listdir(os.path.join(path, crop, "train"))

    if ".DS_Store" in crop_descriptions_temp:
        crop_descriptions_temp.remove(".DS_Store")

    for crop_class in crop_descriptions_temp:
        # Loop through images in each class
        for set in ["train", "test"]:
            for roots, dirs, files in os.walk(
                os.path.join(path, crop, set, crop_class)
            ):
                for file in files:
                    data.append(
                        (
                            crop,
                            set,
                            crop_class,
                            os.path.join(crop, set, crop_class, file),
                        )
                    )

df = pd.DataFrame(
    data, columns=["crop", "set", "crop_description", "file"]
)  # convert to pandas
# df["label_healthy"] = df["crop_description"].apply(lambda x: 1 if x == "Healthy" else 0)

df["label"] = LabelEncoder().fit_transform(df["crop_description"])

# Create label dictionary
crop_descriptions = {}  # initialize dictionary
for crop in crops:
    crop_descriptions[crop] = {}
    unique_pairs = df[["crop_description", "label"]].drop_duplicates()

    # Loop over the rows of the DataFrame
    for _, row in unique_pairs.iterrows():
        # Add the pair to the dictionary
        crop_descriptions[crop][row["label"]] = row["crop_description"]


# Define dataset class
class AugmentedCCMT(Dataset):
    def __init__(self, config, df, transform=None, mode="val"):
        # Set attributes
        # self.image_dir = config.image_dir
        self.image_dir = path
        self.df = df
        self.files = df["file"].values
        self.labels = df["label"].values

        # Define transformation
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get label and file_path for index
        label = self.labels[idx]
        file_path = os.path.join(self.image_dir, self.files[idx])

        try:
            # Read image at file_path with PIL
            image = Image.open(file_path)

            # Apply transformations
            image = self.transform(image)
            # image = augmented["image"]
        # Your existing code here
        except Exception as e:
            print(f"Error loading image at index {idx} for {label}; {e}; {file_path}")
            raise

        return image, label


# Add gaussian blur?
transform_train = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Define metric
def calculate_metric(y, y_pred):
    metric = accuracy_score(y, y_pred)
    return metric


# Training step of a single epoch
# @profile
def train_epoch(dataloader, model, optimizer, config):
    # Training mode
    model.train()

    epoch_loss_long = []
    epoch_labels_long = []
    epoch_output_long = []

    for i, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = images.to(config.device)
        labels = labels.to(config.device)
        epoch_labels_long.extend(labels.detach().cpu().numpy().tolist())

        # Zero gradient
        optimizer.zero_grad()

        # Enable gradients (for the avoidance of doubt)
        with torch.set_grad_enabled(True):
            # Forward pass
            # outputs = model(images)
            outputs = checkpoint(model, images, use_reentrant=False)
            epoch_output_long.extend(outputs.detach().cpu().numpy().tolist())

            # Calculate loss
            loss = config.criterion(outputs, labels)
            epoch_loss_long.append(loss.item())

            # Backpropogation
            loss.backward()

            # Update weights
            optimizer.step()

    epoch_loss = np.mean(epoch_loss_long)

    epoch_labels_pred_long = np.argmax(epoch_output_long, axis=1)
    epoch_accuracy = calculate_metric(epoch_labels_long, epoch_labels_pred_long)

    unique_labels = np.union1d(epoch_labels_long, epoch_labels_pred_long)

    tab = np.zeros((len(unique_labels), len(unique_labels)))

    for i in unique_labels:
        for j in unique_labels:
            tab[i, j] = np.sum((epoch_labels_long == i) & (epoch_labels_pred_long == j))

    return model, epoch_loss, epoch_accuracy, tab


# Validation step of a single epoch
# @profile
def validate_epoch(dataloader, model, config):
    # Evaluation mode
    model.eval()

    epoch_loss_long = []
    epoch_labels_long = []
    epoch_output_long = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = images.to(config.device)
            labels = labels.to(config.device)
            epoch_labels_long.extend(labels.detach().cpu().numpy().tolist())

            # Forward pass
            outputs = model(images)
            epoch_output_long.extend(outputs.detach().cpu().numpy().tolist())

            # Calculate loss
            loss = config.criterion(outputs, labels)
            epoch_loss_long.append(loss.item())

    epoch_loss = np.mean(epoch_loss_long)

    epoch_labels_pred_long = np.argmax(epoch_output_long, axis=1)
    epoch_accuracy = calculate_metric(epoch_labels_long, epoch_labels_pred_long)

    unique_labels = np.union1d(epoch_labels_long, epoch_labels_pred_long)

    tab = np.zeros((len(unique_labels), len(unique_labels)))

    for i in unique_labels:
        for j in unique_labels:
            tab[i, j] = np.sum((epoch_labels_long == i) & (epoch_labels_pred_long == j))

    return epoch_loss, epoch_accuracy, tab


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    # PyTorch seed
    torch.manual_seed(seed)
    
    # CUDA 1 GPU seed
    torch.cuda.manual_seed(seed)
    
    # CUDA multi-GPU seed
    torch.cuda.manual_seed_all(seed)
    
    # Force deterministic operations in cudnn
    torch.backends.cudnn.deterministic = True 
    
    # Disable cudnn auto-tuner
    torch.backends.cudnn.benchmark = False