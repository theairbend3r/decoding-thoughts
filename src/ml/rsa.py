import torch
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from src.ml.model import StimulusClassifier
from src.ml.utils import (
    count_model_params,
    calc_model_frobenius_norm,
    get_latent_emb_per_class,
)


def calculate_rsm(
    model, dataloader, rsm_entity, config, agg_class, plot, class2idx, idx2class
):

    latent_emb_dict = get_latent_emb_per_class(
        model=model,
        dataloader=dataloader,
        agg_class=agg_class,
        class2idx=class2idx,
        idx2class=idx2class,
    )

    if agg_class:
        latent_emb = torch.stack(list(latent_emb_dict.values()), dim=0)
        rsm = torch.nn.functional.cosine_similarity(
            latent_emb.reshape(1, len(class2idx), config.latent_emb_size),
            latent_emb.reshape(len(class2idx), 1, config.latent_emb_size),
            dim=2,
        )

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
            plt.title(f"RSM: {rsm_entity}")

    else:
        latent_emb = torch.cat(list(latent_emb_dict.values()), dim=0)

        rsm = torch.nn.functional.cosine_similarity(
            latent_emb.reshape(1, len(dataloader), config.latent_emb_size),
            latent_emb.reshape(len(dataloader), 1, config.latent_emb_size),
            dim=2,
        )

        # grouped labels for heatmap
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

        if plot:
            plt.figure(figsize=(15, 10))
            sns.heatmap(
                data=rsm,
                cmap="mako",
                xticklabels=final_label_list,
                yticklabels=final_label_list,
            )
            plt.title(f"RSM: {rsm_entity}")

    return latent_emb


def run_rsa(
    fmri_model, fmri_loader, fmri_config, stim_loader, stim_config, class2idx, idx2class
):

    stim_model_norm_list = []
    stim_model_num_param_list = []

    # fMRI model
    # fmri_rsm
    _ = calculate_rsm(
        model=fmri_model,
        dataloader=fmri_loader,
        rsm_entity="fMRI Classifier",
        config=fmri_config,
        agg_class=False,
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
        _ = calculate_rsm(
            model=stim_model,
            dataloader=stim_loader,
            rsm_entity=f"Stimulus Classifier | {model_name.capitalize()}",
            config=stim_config,
            agg_class=False,
            plot=True,
            class2idx=class2idx,
            idx2class=idx2class,
        )

        # append items to list for returning
        stim_model_norm_list.append(calc_model_frobenius_norm(stim_model))
        stim_model_num_param_list.append(count_model_params(stim_model))

    return (
        stim_config.model_names,
        stim_model_norm_list,
        stim_model_num_param_list,
    )


# def calculate_rsm(
#     model_tuple,
#     dataloader_tuple,
#     rsm_entity,
#     config_tuple,
#     agg_class,
#     plot,
#     class2idx,
#     idx2class,
# ):

#     latent_emb_dict_1 = get_latent_emb_per_class(
#         model=model_tuple[0],
#         dataloader=dataloader_tuple[0],
#         agg_class=agg_class,
#         class2idx=class2idx,
#         idx2class=idx2class,
#     )

#     latent_emb_dict_2 = get_latent_emb_per_class(
#         model=model_tuple[1],
#         dataloader=dataloader_tuple[1],
#         agg_class=agg_class,
#         class2idx=class2idx,
#         idx2class=idx2class,
#     )

#     if agg_class:
#         latent_emb_1 = torch.stack(list(latent_emb_dict_1.values()), dim=0)
#         latent_emb_2 = torch.stack(list(latent_emb_dict_2.values()), dim=0)

#         if latent_emb_1.shape != latent_emb_2.shape:
#             raise ValueError("Latent embeddings do not have the same shape.")

#         rsm = torch.nn.functional.cosine_similarity(
#             latent_emb_1.reshape(1, len(class2idx), config_tuple[0].latent_emb_size),
#             latent_emb_2.reshape(len(class2idx), 1, config_tuple[1].latent_emb_size),
#             dim=2,
#         )

#         if plot:
#             plt.figure(figsize=(15, 10))
#             sns.heatmap(
#                 data=rsm,
#                 annot=True,
#                 annot_kws={"fontsize": 14},
#                 cmap="mako",
#                 xticklabels=class2idx.keys(),
#                 yticklabels=class2idx.keys(),
#             )
#             plt.title(f"RSM: {rsm_entity}")

#     else:
#         latent_emb_1 = torch.cat(list(latent_emb_dict_1.values()), dim=0)
#         latent_emb_2 = torch.cat(list(latent_emb_dict_2.values()), dim=0)

#         if latent_emb_1.shape != latent_emb_2.shape:
#             raise ValueError("Latent embeddings do not have the same shape.")

#         rsm = torch.nn.functional.cosine_similarity(
#             latent_emb_1.reshape(
#                 1, len(dataloader_tuple[0]), config_tuple[0].latent_emb_size
#             ),
#             latent_emb_2.reshape(
#                 len(dataloader_tuple[1]), 1, config_tuple[1].latent_emb_size
#             ),
#             dim=2,
#         )

#         # grouped labels for heatmap
#         label_len_list = [len(latent_emb_dict_1[k]) for k in latent_emb_dict_1.keys()]

#         final_label_list = []
#         for i, label_len in enumerate(label_len_list):
#             half_len = label_len // 2
#             list_per_class = ["|" for i in range(label_len)]
#             if len(list_per_class) > 1:
#                 list_per_class[0] = " "
#                 list_per_class[half_len] = idx2class[i]
#                 list_per_class[-1] = " "
#             else:
#                 list_per_class[half_len] = idx2class[i]

#             final_label_list.extend(list_per_class)

#         if plot:
#             plt.figure(figsize=(15, 10))
#             sns.heatmap(
#                 data=rsm,
#                 cmap="mako",
#                 xticklabels=final_label_list,
#                 yticklabels=final_label_list,
#             )
#             plt.title(f"RSM: {rsm_entity}")

#     return rsm


# def run_rsa(
#     fmri_model, fmri_loader, fmri_config, stim_loader, stim_config, class2idx, idx2class
# ):
#     stim_vs_fmri_cosinesim_norm_list = []
#     stim_model_norm_list = []
#     stim_model_num_param_list = []

#     # fMRI model
#     fmri_fmri_rsm = calculate_rsm(
#         model_tuple=(fmri_model, fmri_model),
#         dataloader_tuple=(fmri_loader, fmri_loader),
#         rsm_entity="fMRI vs fMRI",
#         config_tuple=(fmri_config, fmri_config),
#         agg_class=False,
#         plot=True,
#         class2idx=class2idx,
#         idx2class=idx2class,
#     )

#     # Stimulus model
#     for model_name in tqdm(stim_config.model_names):

#         stim_model = StimulusClassifier(
#             num_classes=len(class2idx), model_name=model_name
#         )
#         stim_model.load_state_dict(
#             torch.load(
#                 f"./../models/stimulus_classifier/stim_classifier_model_{model_name}.pth"
#             )
#         )
#         stim_model.eval()

#         stim_stim_rsm = calculate_rsm(
#             model_tuple=(stim_model, stim_model),
#             dataloader_tuple=(stim_loader, stim_loader),
#             rsm_entity=f"Stimulus vs Stimulus | {model_name.capitalize()}",
#             config_tuple=(stim_config, stim_config),
#             agg_class=False,
#             plot=True,
#             class2idx=class2idx,
#             idx2class=idx2class,
#         )

#         stim_fmri_rsm = calculate_rsm(
#             model_tuple=(stim_model, fmri_model),
#             dataloader_tuple=(stim_loader, fmri_loader),
#             rsm_entity=f"Stimulus vs fMRI | {model_name.capitalize()}",
#             config_tuple=(stim_config, fmri_config),
#             agg_class=False,
#             plot=True,
#             class2idx=class2idx,
#             idx2class=idx2class,
#         )

#         # print(
#         #     f"Norm of cosine similarity between stimulus ({model_name}) and fmri embeddings = {stim_fmri_cosinesim_norm}"
#         # )

#         stim_model_norm_list.append(calc_model_frobenius_norm(stim_model))
#         stim_model_num_param_list.append(count_model_params(stim_model))
#         # stim_vs_fmri_cosinesim_norm_list.append(stim_fmri_cosinesim_norm)

#     return (
#         stim_config.model_names,
#         stim_vs_fmri_cosinesim_norm_list,
#         stim_model_norm_list,
#         stim_model_num_param_list,
#     )
