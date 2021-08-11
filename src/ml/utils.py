import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


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


def plot_loss_acc_curves(loss_stats: dict, acc_stats: dict):
    """Plot loss and accuracy curves.

    Parameters
    ----------
    loss_stats:
        A dictionary with loss values and keys = "train" and "val".
    acc_stats:
        A dictionary with accuracy values and keys = "train" and "val".
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
    axes[0].set_title("Accuracy (train/loss)")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Values")

    sns.lineplot(
        data=train_val_loss_df, x="epochs", y="value", hue="variable", ax=axes[1]
    )
    axes[1].set_title("Loss (train/loss)")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Values")

    plt.suptitle("Loss-Accuracy Curves", fontsize=18)


def generate_score_report(y_true: list, y_pred: list, idx2class):
    """Generates a score report.

    Prints the classification report and plots a confusion matrix.

    Parameters
    ----------
    y_true: list
        A 1-d numpy array or list of true values.
    y_pred: list
        A 1-d numpy array or list of predicted values.
    """
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred))
    print("\n\n")

    df = pd.DataFrame(confusion_matrix(y_true, y_pred)).rename(
        columns=idx2class, index=idx2class
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt="g", cbar=False, annot_kws={"fontsize": 14})
    plt.xlabel("Pred Output")
    plt.ylabel("True Output")
    plt.title("Confusion Matrix")
