# Libraries
import tools.pest_classification as pest

import json
import matplotlib.pyplot as plt
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

    # Save index to name file
    with open(os.path.join(path_out, "index_to_name.json"), "w") as f:
        json.dump(pest.crop_descriptions["Maize"], f)

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

    # Initialize history plots
    fig_all, ax_all = plt.subplots(2, 2, figsize=(10, 10))
    ax_all[0, 0].set_title("Training loss")
    ax_all[0, 0].set_xlabel("Epoch")
    ax_all[0, 0].set_ylabel("Loss")
    ax_all[0, 1].set_title("Training accuracy")
    ax_all[0, 1].set_xlabel("Epoch")
    ax_all[0, 1].set_ylabel("Accuracy")
    ax_all[1, 0].set_title("Validation loss")
    ax_all[1, 0].set_xlabel("Epoch")
    ax_all[1, 0].set_ylabel("Loss")
    ax_all[1, 1].set_title("Validation accuracy")
    ax_all[1, 1].set_xlabel("Epoch")
    ax_all[1, 1].set_ylabel("Accuracy")

    fig_avg, ax_avg = plt.subplots(1, 2, figsize=(10, 5))
    ax_avg[0].set_title("Loss")
    ax_avg[0].set_xlabel("Epoch")
    ax_avg[0].set_ylabel("Loss")
    ax_avg[1].set_title("Accuracy")
    ax_avg[1].set_xlabel("Epoch")
    ax_avg[1].set_ylabel("Accuracy")
    
    # Process loss and accuracy histories
    k = 0

    for sample in ["train", "valid"]:
        for stat in ["loss", "accuracy"]:
            k_1 = k // 2
            k_2 = k % 2
            k += 1

            avg = []

            for fold in range(config_dict["n_folds"]):
                with open(os.path.join(path_out, f"fold_{fold}_{sample}_{stat}_history.txt"), "r") as file:
                    history = np.loadtxt(file, delimiter=",")

                # Plot loss and accuracy histories
                ax_all[k_1, k_2].plot(history, label=f"Fold {fold}")
                avg.append(history)
    
            ax_all[k_1, k_2].legend()

            avg = np.mean(avg, axis=0)
            ax_avg[k_2].plot(avg, label=f"{sample.capitalize()}")
            ax_avg[k_2].legend()

    # Save figures
    fig_all.tight_layout()
    fig_all.savefig(os.path.join(path_out, "all_history.png"))
    fig_avg.tight_layout()
    fig_avg.savefig(os.path.join(path_out, "avg_history.png"))
    
    # Load index to name file
    with open(os.path.join(path_out, "index_to_name.json"), "r") as file:
        index_to_name = json.load(file)
    labels = ["[" + index_to_name[str(i)] + "]" for i in range(len(index_to_name))]

    with open(os.path.join(path_out, "summary.typ"), "w") as file:
        file.write(
            f"""
            #import "@preview/tablex:0.0.8": tablex, cellx, hlinex, vlinex

            Cross-validation summary: `{name}`\n

            Configuration: {config_dict}
            """
            )

        # All epoch history plots
        file.write(
            """
            #figure(caption: "Loss and accuracy histories")[
                #image(\"all_history.png\")
            ]\n
            """
        )

        # Average epoch history plots
        file.write(
            """
            #figure(caption: "Average loss and accuracy histories")[
                #image(\"avg_history.png\")
            ]\n
            """
        )

        # Load tabulations
        for sample in ["train", "valid"]:
            for fold in range(config_dict["n_folds"]):
                with open(os.path.join(path_out, f"fold_{fold}_epoch_{config_dict['num_epochs'] - 1}_tab_{sample}.txt"), "r") as file_tab:
                    tab = np.loadtxt(file_tab, delimiter=",")

                    sum_pred = np.diag(tab) / np.sum(tab, axis=1)
                    sum_label = np.diag(tab) / np.sum(tab, axis=0)

                    tab_str = np.array([["%.0f" % num for num in row] for row in tab])
                    sum_pred_str = np.array(['%.3f' % num for num in sum_pred])
                    sum_label_str = np.array(['%.3f' % num for num in sum_label])
                    sum_label_str = np.append(sum_label_str, "[]")

                    tab_str = np.vstack((np.column_stack((tab_str, sum_pred_str)), sum_label_str))
                    tab_str = np.vstack((labels + ["vlinex(),[],hlinex()"], tab_str))
                    tab_str = np.column_stack((["[],vlinex()"] + labels + ["hlinex(),[]"], tab_str, np.repeat("", tab_str.shape[0])))

                    file.write(
                        f"""
                        #figure(caption: "Tabulation, Fold {fold}, {sample}", tablex(
                            columns: 9,
                            align: center + horizon,
                            auto-vlines: false,
                            auto-hlines: false,
                            header-rows: 1,
                        """
                    )

                    for row in tab_str:
                        file.write(",".join(row) + "\n")

                    file.write(
                        """
                        ))
                        """
                    )


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