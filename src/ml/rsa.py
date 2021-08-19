"""
Functions for representation similarity analysis.
"""
import gc
import numpy as np

import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from src.ml.model import StimulusClassifier
from src.ml.utils import (
    count_model_params,
    calc_model_frobenius_norm,
    get_latent_emb_per_class,
)


def calculate_rsm(
    model: nn.Module,
    dataloader: DataLoader,
    plot_title: str,
    config: dict,
    agg_class: bool,
    plot: bool,
    class2idx: dict,
    idx2class: dict,
) -> torch.tensor:

    """
    Calculates (and plots) the representation similarity matrices using
    embeddings from a train neural network. Uses cosine similarity.


    Parameters
    ----------
    model:
        PyTorch model with trained weights.
    dataloader:
        Dataloader object.
    plot_title:
        Plot title.
    config:
        Configuration object (for fmri or stmim).
    agg_class:
        If True, embeddings are averaged before RSM calculation.
    plot:
        If True, a heatmap is plotted.
    class2idx:
        A mapping between class and it's corresponding integer idx.
    idx2class:
        Reverse mapping of class2idx.


    Returns
    -------
    torch.tensor
        The representation similarity matrix.
    """

    # get dictionary of latent embedding with classes as keys and a list
    # of tensor as values.
    latent_emb_dict = get_latent_emb_per_class(
        model=model,
        dataloader=dataloader,
        agg_class=agg_class,
        class2idx=class2idx,
        idx2class=idx2class,
    )

    # If agg_class is True, then take the mean along the list for all tensors
    # to obtain a single embedding per class.
    if agg_class:
        latent_emb = torch.stack(list(latent_emb_dict.values()), dim=0)

        rsm = torch.nn.functional.cosine_similarity(
            latent_emb.reshape(1, len(class2idx), config.latent_emb_size),
            latent_emb.reshape(len(class2idx), 1, config.latent_emb_size),
            dim=2,
        )

        # plot the representational similarity matrix
        if plot:
            plt.figure(figsize=(15, 10))
            sns.heatmap(
                data=rsm,
                annot=True,
                annot_kws={"fontsize": 14},
                cmap="mako",
                xticklabels=class2idx.keys(),
                yticklabels=class2idx.keys(),
            )
            plt.title(f"RSM: {plot_title}")

    else:
        latent_emb = torch.cat(list(latent_emb_dict.values()), dim=0)

        rsm = torch.nn.functional.cosine_similarity(
            latent_emb.reshape(1, len(dataloader), config.latent_emb_size),
            latent_emb.reshape(len(dataloader), 1, config.latent_emb_size),
            dim=2,
        )

        # group labels for heatmap xticklabels and yticklabels.
        label_len_list = [len(latent_emb_dict[k]) for k in latent_emb_dict.keys()]

        final_label_list = []
        for i, label_len in enumerate(label_len_list):
            half_len = label_len // 2
            list_per_class = ["|" for i in range(label_len)]
            if len(list_per_class) > 1:
                list_per_class[0] = " "
                list_per_class[half_len] = idx2class[i]
                list_per_class[-1] = " "
            else:
                list_per_class[half_len] = idx2class[i]

            final_label_list.extend(list_per_class)

        # plot the representational similarity matrix
        if plot:
            plt.figure(figsize=(15, 10))
            sns.heatmap(
                data=rsm,
                cmap="mako",
                xticklabels=final_label_list,
                yticklabels=final_label_list,
            )
            plt.title(f"RSM: {plot_title}")

    return rsm


def run_rsa(
    fmri_model: nn.Module,
    fmri_loader: DataLoader,
    fmri_config: dict,
    stim_loader: DataLoader,
    stim_config: dict,
    class2idx: dict,
    idx2class: dict,
) -> tuple:
    """
    Parameters
    ----------
    fmri_model:
        Model trained on fMRI dataset.
    fmri_loader:
        Dataloader for fMRI dataset.
    fmri_config:
        Configuration object for fMRI model.
    stim_loader:
        Dataloader for stimulus dataset.
    stim_config:
        Configuration object for stimulus model.
    class2idx:
        A mapping between class and it's corresponding integer idx.
    idx2class:
        Reverse mapping of class2idx.

    Returns
    -------
    tuple
        Returns a tuple of lists that contain model names,
        model norms, and number of parameters in the model.
        (stim_config.model_names, stim_model_norm_list, stim_model_num_param_list)
    """

    stim_model_norm_list = []
    stim_model_num_param_list = []

    # fMRI model
    # fmri_rsm
    fmri_rsm = calculate_rsm(
        model=fmri_model,
        dataloader=fmri_loader,
        plot_title="fMRI Classifier",
        config=fmri_config,
        agg_class=False,
        plot=True,
        class2idx=class2idx,
        idx2class=idx2class,
    )

    _ = calculate_rsm(
        model=fmri_model,
        dataloader=fmri_loader,
        plot_title="fMRI Classifier | Aggregated Class | ",
        config=fmri_config,
        agg_class=True,
        plot=True,
        class2idx=class2idx,
        idx2class=idx2class,
    )

    # Stimulus model
    for model_name in tqdm(stim_config.model_names):

        # load model
        stim_model = StimulusClassifier(
            num_classes=len(class2idx), model_name=model_name
        )
        stim_model.load_state_dict(
            torch.load(
                f"./../models/stimulus_classifier/stim_classifier_model_{model_name}.pth",
                map_location="cpu",
            )
        )
        stim_model.eval()

        # calculate rsm matrix for stimulus embeddings
        # stim_rsm
        stim_rsm = calculate_rsm(
            model=stim_model,
            dataloader=stim_loader,
            plot_title=f"Stimulus Classifier | {model_name.capitalize()}",
            config=stim_config,
            agg_class=False,
            plot=True,
            class2idx=class2idx,
            idx2class=idx2class,
        )

        _ = calculate_rsm(
            model=stim_model,
            dataloader=stim_loader,
            plot_title=f"Stimulus Classifier | Aggregated Class | {model_name.capitalize()}",
            config=stim_config,
            agg_class=True,
            plot=True,
            class2idx=class2idx,
            idx2class=idx2class,
        )

        print(
            f"Correlation betweeb {model_name} and fMRI - {np.corrcoef(np.array(stim_rsm).flatten(), np.array(fmri_rsm).flatten())[0, 1]}"
        )

        # append items to list for returning
        stim_model_norm_list.append(calc_model_frobenius_norm(stim_model))
        stim_model_num_param_list.append(count_model_params(stim_model))

        del stim_model
        gc.collect()

    return (
        stim_config.model_names,
        stim_model_norm_list,
        stim_model_num_param_list,
    )
