#!/usr/bin/env python3

from typing import Tuple

import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple


def attribution_localization(
    attributions: Tuple[torch.Tensor, ...],
    segmentation_masks: Tuple[torch.Tensor, ...],
    positive_attributions: bool = True,
    weighted: bool = False,
    is_multi_target: bool = False,
    return_dict: bool = False,
) -> torch.Tensor:
    """
    Implementation of the Attribution Localization by Kohlbrenner et al., 2020. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library.

    Attribution Localization implements the ratio of positive attributions within the target to the overall
    attribution. High scores are desired, as it means, that the positively attributed pixels belong to the
    targeted object class.

    References:
        1) Max Kohlbrenner et al., "Towards Best Practice in Explaining Neural Network Decisions with LRP."
        IJCNN (2020): 1-7.

    Args:
        attributions (Tuple[Tensor,...]): A tuple of tensors representing attributions of separate inputs. Each
            tensor in the tuple has shape (batch_size, num_features).
        segmentation_masks (Tuple[Tensor,...]): A tuple of boolean mask tensors
            representing the desired segmented region for for each input attribution.
        weighted (bool, optional): If True, the metric is weighted by the ratio of the total mask size and the
            size of segmented region.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
            with keys as the metric names and values as the corresponding metric outputs.
            Default is False.

    Returns:
        Tensor: A tensor of scalar complexity scores per
                input example. The first dimension is equal to the
                number of examples in the input batch and the second
                dimension is one.
    Examples::
        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> saliency = Saliency(net)
        >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> baselines = torch.zeros(2, 3, 32, 32)
        >>> # Computes saliency maps for class 3.
        >>> attribution = saliency.attribute(input, target=3)
        >>> # define a perturbation function for the input

        >>> # Computes the monotonicity correlation and non-sensitivity scores for saliency maps
        >>> attribution_localization_score = attribution_localization(attribution, feature_mask)
    """
    if is_multi_target:
        isinstance(
            attributions, list
        ), "attributions must be a list of tensors or list of tuples of tensors"
        attribution_localization_scores = [
            attribution_localization(
                attributions=a,
                segmentation_masks=segmentation_masks,
                positive_attributions=positive_attributions,
                weighted=weighted,
                return_dict=False,
            )
            for a in attributions
        ]
        if return_dict:
            return {
                "attribution_localization_score": attribution_localization_scores,
            }
        return attribution_localization_scores

    with torch.no_grad():
        is_attributions_tuple = _is_tuple(attributions)
        attributions = _format_tensor_into_tuples(attributions)
        segmentation_masks = _format_tensor_into_tuples(segmentation_masks)
        assert (
            segmentation_masks[0].dtype == torch.bool
        ), "Segmentation mask must be of type bool."
        assert (
            len(segmentation_masks) == len(attributions)
            and segmentation_masks[0].shape == attributions[0].shape
        ), "Segmentation mask must have the same shape as the attributions."

        if positive_attributions:
            attributions = tuple(
                torch.clamp(attribution, min=0) for attribution in attributions
            )

        bsz = attributions[0].shape[0]
        attribution_localization_scores = tuple(
            (attribution * mask).view(bsz, -1).sum(dim=1)
            / attribution.view(bsz, -1).sum(dim=1)
            for attribution, mask in zip(attributions, segmentation_masks)
        )
        mask_size_ratios = tuple(
            mask.numel() / mask.contiguous().view(bsz, -1).sum(dim=1)
            for mask in segmentation_masks
        )

        if weighted:
            attribution_localization_scores = tuple(
                score * mask_size_ratio
                for score, mask_size_ratio in zip(
                    attribution_localization_scores, mask_size_ratios
                )
            )

        attribution_localization_scores = _format_output(
            is_attributions_tuple, attribution_localization_scores
        )
        if return_dict:
            {
                "attribution_localization_score": attribution_localization_scores,
            }
        return attribution_localization_scores
