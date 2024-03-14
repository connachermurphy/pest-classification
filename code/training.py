# Libraries
import pest_classification as pest

import json
# import matplotlib.pyplot as plt
# from memory_profiler import profile
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
import time
import timm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace

path = os.path.expanduser("~/projects/pest-classification/code/training_results")


def train(train_dataloader, model, optimizer, config):
    # Output path
    path_out = os.path.join(path, config.name)

    # Store loss and accuracy history
    loss_history = []
    accuracy_history = []

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}")
        
        model, train_loss, train_accuracy, train_tab = pest.train_epoch(
            train_dataloader, model, optimizer, config
        )

        loss_history.append(train_loss)
        accuracy_history.append(train_accuracy)

        print(f"Training: loss = {train_loss}, accuracy = {train_accuracy}")
        
        # Save tabulations for last epoch
        if epoch == config.num_epochs - 1:
            np.savetxt(os.path.join(path_out, f"epoch_{epoch}_tab_train.txt"), train_tab, delimiter=",", fmt="%d")

        print("\n")

    return model, loss_history, accuracy_history


# @profile
def training_run(config):
    """
    Perform a training run
    """
    print(f"Starting training run: {config.name}")

    start = time.time()

    # Output path
    path_out = os.path.join(path, config.name)

    # Abort if directory already exists
    if os.path.isdir(path_out):
        print("Directory already exists, aborting")
        return
    
    # Create directory
    os.mkdir(path_out)

    # Report configuration
    config_dict = {key: value for key, value in vars(config).items() if not key.startswith("__")}

    with open(os.path.join(path_out, "config.json"), "w") as json_file:
        json.dump(config_dict, json_file)

    # Set random seed
    pest.set_seed(config.seed)

    # Grab training observations from images df
    df_all = pest.df
    df = df_all[df_all["set"] == "train"]
    # df = df_all[df_all["set"] == "train"].sample(512)
    df = df.reset_index(drop=True)

    # Architecture
    config.num_classes = len(pest.crop_descriptions["Maize"])
    config.backbone = "resnet18"

    # Select GPU if available
    print("CUDA availability:", torch.cuda.is_available())
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader
    train_dataset = pest.AugmentedCCMT(config, df, transform=pest.transform_train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Initialize (pre-trained) model
    model = timm.create_model(
        config.backbone, pretrained=True, num_classes=config.num_classes
    )
    model.to(config.device)

    # Specify loss function (CM: move this to outer loop?)
    config.criterion = nn.CrossEntropyLoss()

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0)

    # Call training function
    (
        model,
        loss_history,
        accuracy_history,
    ) = train(train_dataloader, model, optimizer, config)

    # Save loss and accuracy histories
    np.savetxt(os.path.join(path_out, "loss_history.txt"), loss_history, delimiter=",")
    np.savetxt(os.path.join(path_out, "accuracy_history.txt"), accuracy_history, delimiter=",")

    end = time.time()
    elapsed = (end - start) / 60
    print("Training run completed in {:,.0f} minutes".format(elapsed))

    # Save model artifacts
    print("Saving model artifacts")
    torch.save(model, os.path.join(path_out, f"{config.name}.pth"))
    with open(os.path.join(path_out, "index_to_name.json"), "w") as f:
        json.dump(pest.crop_descriptions["Maize"], f)

    print("Done :)")

if __name__ == "__main__":
    """
    Default configuration
    """
    config = SimpleNamespace(**{})
    config.name = "test"
    config.seed = 123
    config.batch_size = 128
    config.num_epochs = 1
    config.lr = 1e-4

    training_run(config)