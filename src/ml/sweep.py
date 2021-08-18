import torch.nn as nn
import torch.optim as optim
from src.ml.test import test_model
from src.ml.model import StimulusClassifier
from src.ml.utils import plot_loss_acc_curves, generate_score_report


def train_all_stim_models(
    train_loader,
    test_loader,
    model_name,
    config,
    train_model,
    class2idx,
    idx2class,
    device,
):
    model = StimulusClassifier(num_classes=len(class2idx), model_name=model_name)
    model.to(device)

    epochs = config.epochs
    criterion = nn.CrossEntropyLoss()  # weight=get_class_weights(y_train).to(device)
    optimizer = optim.Adam(
        (param for param in model.parameters() if param.requires_grad is True),
        lr=config.learning_rate,
    )

    loss_stats, acc_stats = train_model(
        model=model,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_model_path=f"./../models/stimulus_classifier/stim_classifier_model_{model_name}.pth",
    )

    plot_loss_acc_curves(
        loss_stats=loss_stats, acc_stats=acc_stats, model_name=model_name
    )
    y_true_list, y_pred_list = test_model(
        model=model, test_loader=test_loader, device=device
    )
    generate_score_report(
        y_true=y_true_list,
        y_pred=y_pred_list,
        idx2class=idx2class,
        model_name=model_name,
    )

    del model
