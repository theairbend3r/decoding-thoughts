import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
from src.ml.utils import calc_multi_acc, print_log


def train_model(
    model: nn.Module,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    criterion,
    device: str,
    save_model_path: str,
):
    """Training loop.

    Parameters
    ----------
    model:
        PyTorch model object.
    epochs:
        Integer that denotes the number of training iterations.
    train_loader:
        Dataloader for train dataset.
    val_loader
        Dataloader for validation dataset.
    optimizer:
        PyTorch optimizer.
    criterion:
        PyTorch loss function.
    device:
        CUDA identifier.
    save_model_path:
        Path to save the trained model weights.

    Returns
    -------
    tuple
        A tuple of dictionaries (loss_stats, acc_stats)
    """
    if type(epochs) != int:
        raise ValueError("epochs has to be an integer.")

    acc_stats = {"train": [], "val": []}
    loss_stats = {"train": [], "val": []}

    for e in tqdm(range(1, epochs + 1)):

        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for x_train_batch, y_train_batch in train_loader:
            x_train_batch, y_train_batch = (
                x_train_batch.to(device),
                y_train_batch.to(device),
            )

            optimizer.zero_grad()

            y_train_pred = model(x_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = calc_multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        val_epoch_loss = 0
        val_epoch_acc = 0
        with torch.no_grad():
            model.eval()
            for x_val_batch, y_val_batch in val_loader:
                x_val_batch, y_val_batch = (
                    x_val_batch.to(device),
                    y_val_batch.to(device),
                )

                y_val_pred = model(x_val_batch).squeeze()

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = calc_multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        avg_train_epoch_loss = train_epoch_loss / len(train_loader)
        avg_train_epoch_acc = train_epoch_acc / len(train_loader)
        avg_val_epoch_loss = val_epoch_loss / len(val_loader)
        avg_val_epoch_acc = val_epoch_acc / len(val_loader)

        loss_stats["train"].append(avg_train_epoch_loss)
        loss_stats["val"].append(avg_val_epoch_loss)
        acc_stats["train"].append(avg_train_epoch_acc)
        acc_stats["val"].append(avg_val_epoch_acc)

        print_log(
            e,
            epochs,
            avg_train_epoch_loss,
            avg_val_epoch_loss,
            avg_train_epoch_acc,
            avg_val_epoch_acc,
        )

    if save_model_path:
        torch.save(model.state_dict(), save_model_path)
    return loss_stats, acc_stats
