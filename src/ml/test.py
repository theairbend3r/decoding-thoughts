import torch
from tqdm.notebook import tqdm


def test_model(model, test_loader, device):
    """Testing loop.

    model:
        PyTorch model object.
    train_loader:
        Dataloader for test dataset.
    device:
        CUDA identifier.

    Returns
    -------
    tuple
        A tuple of lists (y_true_list, y_pred_list)
    """
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_test_pred = model(x_batch)
            _, y_pred_tag = torch.max(y_test_pred, dim=1)

            for i in y_pred_tag.squeeze().cpu().numpy().tolist():
                y_pred_list.append(i)

            for i in y_batch.squeeze().cpu().numpy().tolist():
                y_true_list.append(i)

    return y_true_list, y_pred_list
