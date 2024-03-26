# Libraries
import tools.pest_classification as pest

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

path = os.path.expanduser("~/projects/pest-classification/validation_results")


def validate(train_dataloader, valid_dataloader, model, optimizer, config, fold):
    # Output path
    path_out = os.path.join(path, config.name)

    # Store loss and accuracy history
    train_loss_history = []
    train_accuracy_history = []
    valid_loss_history = []
    valid_accuracy_history = []

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}")
        print("Training...")
        model, train_loss, train_accuracy, train_tab = pest.train_epoch(
            train_dataloader, model, optimizer, config
        )

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        print(f"Training: loss = {train_loss}, accuracy = {train_accuracy}")
        print("\n")

        print("Validating...")
        valid_loss, valid_accuracy, valid_tab = pest.validate_epoch(
            valid_dataloader, model, config
        )

        valid_loss_history.append(valid_loss)
        valid_accuracy_history.append(valid_accuracy)

        print(f"Validation: loss = {valid_loss}, accuracy = {valid_accuracy}")

        # Save tabulations for last epoch
        if epoch == config.num_epochs - 1:
            np.savetxt(os.path.join(path_out, f"fold_{fold}_epoch_{epoch}_tab_train.txt"), train_tab, delimiter=",", fmt="%d")
            np.savetxt(os.path.join(path_out, f"fold_{fold}_epoch_{epoch}_tab_valid.txt"), valid_tab, delimiter=",", fmt="%d")

        print("\n")

    return (
        train_loss_history,
        train_accuracy_history,
        valid_loss_history,
        valid_accuracy_history,
    )


# @profile
def cross_validation(config):
    """
    Perform a cross-validation experiment
    """
    print(f"Starting cross-validation experiment: {config.name}")

    start_all = time.time()

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

    # Add folds to df
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True)
    for fold, (train_index, val_index) in enumerate(skf.split(df, df.label)):
        df.loc[val_index, "fold"] = fold

    # Architecture
    config.num_classes = len(pest.crop_descriptions["Maize"])
    config.backbone = "resnet18"

    # Select GPU if available
    print("CUDA availability:", torch.cuda.is_available())
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cross-validation loop
    for fold in range(config.n_folds):
        print(f"Fold {fold}")

        start_fold = time.time()

        # Split into training and validation sets
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        valid_df = df[df["fold"] == fold].reset_index(drop=True)

        train_dataset = pest.AugmentedCCMT(config, train_df, transform=pest.transform_train)
        valid_dataset = pest.AugmentedCCMT(config, valid_df)

        # Dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
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
            train_loss_history,
            train_accuracy_history,
            valid_loss_history,
            valid_accuracy_history,
        ) = validate(train_dataloader, valid_dataloader, model, optimizer, config, fold)

        # Save loss and accuracy histories
        np.savetxt(os.path.join(path_out, f"fold_{fold}_train_loss_history.txt"), train_loss_history, delimiter=",")
        np.savetxt(os.path.join(path_out, f"fold_{fold}_train_accuracy_history.txt"), train_accuracy_history, delimiter=",")
        np.savetxt(os.path.join(path_out, f"fold_{fold}_valid_loss_history.txt"), valid_loss_history, delimiter=",")
        np.savetxt(os.path.join(path_out, f"fold_{fold}_valid_accuracy_history.txt"), valid_accuracy_history, delimiter=",")

        # plt.plot(train_loss_history, label="Training")
        # plt.plot(valid_loss_history, label="Validation")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig(os.path.join(path_out, f"fold_{fold}_loss.png"))
        # plt.close()

        # plt.plot(train_accuracy_history, label="Training")
        # plt.plot(valid_accuracy_history, label="Validation")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.legend()
        # plt.savefig(os.path.join(path_out, f"fold_{fold}_accuracy.png"))
        # plt.close()

        end_fold = time.time()
        elapsed_fold = (end_fold - start_fold) / 60
        print("Fold {} completed in {:,.0f} minutes".format(fold, elapsed_fold))
        print("\n")

    end_all = time.time()
    elapsed_all = (end_all - start_all) / 60
    print("Experiment completed in {:,.0f} minutes".format(elapsed_all))


def summarize(name):
    """
    Summarize results from cross-validation experiment
    """

    # Output path
    path_out = os.path.join(path, name)

    # Abort if directory does not exist
    if not os.path.isdir(path_out):
        print("Directory does not exist, aborting")
        return
    
    # Load configuration
    with open(os.path.join(path_out, "config.json"), "r") as json_file:
        config_dict = json.load(json_file)

    # Load loss and accuracy histories
    print(config_dict)


if __name__ == "__main__":
    """
    Default configuration
    """
    config = SimpleNamespace(**{})
    config.name = "test"
    config.seed = 123
    config.batch_size = 256
    config.num_epochs = 4
    config.n_folds = 5
    config.lr = 1e-4

    cross_validation(config)