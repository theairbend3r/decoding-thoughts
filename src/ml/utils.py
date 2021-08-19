"""
Utility functions for machine learning operations.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def calc_multi_acc(y_pred: torch.tensor, y_test: torch.tensor) -> float:
    """
    Parameters
    ----------
    y_pred:
        A tensor of predicted values.
    y_test:
        A tensor of actual values

    Returns
    -------
    float
        Accuracy percentage.

    """
    y_pred_softmax = torch.softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def print_log(
    e,
    epochs,
    avg_train_epoch_loss,
    avg_val_epoch_loss,
    avg_train_epoch_acc,
    avg_val_epoch_acc,
):
    """
    Print training logs.
    """
    print(
        f"Epoch {e+0:02}/{epochs}: | Train Loss: {avg_train_epoch_loss:.5f} | Val Loss: {avg_val_epoch_loss:.5f} | Train Acc: {avg_train_epoch_acc:.3f}% | Val Acc: {avg_val_epoch_acc:.3f}%"
    )


def plot_loss_acc_curves(loss_stats: dict, acc_stats: dict, model_name: str):
    """Plot loss and accuracy curves.

    Parameters
    ----------
    loss_stats:
        A dictionary with loss values and keys = "train" and "val".
    acc_stats:
        A dictionary with accuracy values and keys = "train" and "val".
    model_name:
        Name of model.
    """
    train_val_acc_df = (
        pd.DataFrame.from_dict(acc_stats)
        .reset_index()
        .melt(id_vars=["index"])
        .rename(columns={"index": "epochs"})
    )
    train_val_loss_df = (
        pd.DataFrame.from_dict(loss_stats)
        .reset_index()
        .melt(id_vars=["index"])
        .rename(columns={"index": "epochs"})
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 10))
    sns.lineplot(
        data=train_val_acc_df, x="epochs", y="value", hue="variable", ax=axes[0]
    )
    axes[0].set_title("Accuracy (train/val)")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")

    sns.lineplot(
        data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]
    )
    axes[1].set_title("Loss (train/val)")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")

    plt.suptitle(f"Loss-Accuracy Curves | {model_name}", fontsize=18)


def generate_score_report(y_true: list, y_pred: list, idx2class: dict, model_name: str):
    """Generates a score report.

    Prints the classification report and plots a confusion matrix.

    Parameters
    ----------
    y_true:
        A 1-d numpy array or list of true values.
    y_pred:
        A 1-d numpy array or list of predicted values.
    model_name:
        Name of model.
    """
    print(f"Classification Report | {model_name}:\n")
    print(classification_report(y_true, y_pred))
    print("\n\n")

    df = pd.DataFrame(confusion_matrix(y_true, y_pred)).rename(
        columns=idx2class, index=idx2class
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt="g", cbar=False, annot_kws={"fontsize": 14})
    plt.xlabel("Pred Output")
    plt.ylabel("True Output")
    plt.title(f"Confusion Matrix | {model_name}")


def get_latent_emb_per_class(
    model: nn.Module,
    dataloader: DataLoader,
    agg_class: bool,
    class2idx: dict,
    idx2class: dict,
) -> dict:
    """
    Get latent embedding divided by output class labels.

    Parameters
    ----------
    model:
        Train torch model.
    dataloader:
        Dataloader object.
    agg_class:
        Averages across tensors of the same class.
    class2idx:
        Maps class to integers idx.
    idx2class:
        Maps integer idx to class.

    Returns
    -------
    dict
        A dictionary with keys as output labels and values
        as a list of tensors.
    """

    # init empty dictionary for hidden emb per class.
    latent_emb_per_class_dict = {k: [] for k in class2idx.keys()}

    # populate the dict.
    with torch.no_grad():
        for x, y in dataloader:
            latent_emb = model.get_latent_rep(x).squeeze()
            latent_emb_per_class_dict[idx2class[y.item()]].append(latent_emb)

    # get mean embedding value per class.
    if agg_class:
        latent_emb_per_class_dict = {
            k: torch.stack(latent_emb_per_class_dict[k], dim=0).mean(dim=(0))
            for k in latent_emb_per_class_dict.keys()
        }

    # get all embedding value per class.
    else:
        latent_emb_per_class_dict = {
            k: torch.stack(latent_emb_per_class_dict[k], dim=0)
            for k in latent_emb_per_class_dict.keys()
        }

    return latent_emb_per_class_dict


def get_class_weights(y_data: np.ndarray) -> torch.tensor:
    """
    Get balanced weights per class for cross-entropy loss.

    Parameters
    ----------
    y_data:
        Numpy array of output class labels.

    Returns
    -------
    torch.tensor
        A tensor with class weights.
    """

    class_weights = torch.tensor(
        compute_class_weight(
            class_weight="balanced", classes=np.unique(y_data), y=y_data
        ),
        dtype=torch.float,
    )

    return class_weights


def count_model_params(model: nn.Module) -> int:
    """
    Count the number of parameters in a model.

    Parameters
    ----------
    model:
        Pytorch model.

    Returns
    -------
    int
        Number of parameters in the model.
    """

    return sum(p.numel() for p in model.parameters())


def calc_model_frobenius_norm(model: nn.Module) -> float:
    """
    Calculate the frobenius norm of model parameters.

    Parameters
    ----------
    model:
        Pytorch model.

    Returns
    -------
    float
        Frobenius norm.
    """
    norm = 0.0
    for param in model.parameters():
        norm += torch.sum(param ** 2)

    norm = norm ** 0.5

    return norm.item()
