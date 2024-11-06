#!/usr/bin/env python3

from typing import List, Tuple, Union

import torch
from torch.distributions import Categorical

from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _reduce_tensor_with_indices,
    _tuple_tensors_to_tensors,
    _validate_feature_mask,
)


def complexity_entropy(
    attributions: Union[Tuple[torch.Tensor, ...], List[Tuple[torch.Tensor, ...]]],
    is_multi_target: bool = False,
    return_dict: bool = False,
) -> torch.Tensor:
    """
    Implementation of Complexity metric by Bhatt et al., 2020. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library.

    Complexity of attributions is defined as the entropy of the fractional contribution of feature x_i to the total
    magnitude of the attribution. A complex explanation is one that uses all features in its explanation to explain
    some decision. Even though such an explanation may be faithful to the model output, if the number of features is
    too large it may be too difficult for the user to understand the explanations, rendering it useless. Smaller value
    of complexity indicates that the explanation is simple and uses fewer features to explain the decision.

    References:
        1) Umang Bhatt et al.: "Evaluating and aggregating
        feature-based model explanations." IJCAI (2020): 3016-3022.

    Args:
        attributions (Tuple[Tensor,...]): A tuple of tensors representing attributions of separate inputs. Each
            tensor in the tuple has shape (batch_size, num_features).
        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class.
                Default is False.
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
        >>> complexity_scores = complexity(attribution)
    """
    if is_multi_target:
        isinstance(
            attributions, list
        ), "attributions must be a list of tensors or list of tuples of tensors"
        complexity_score = [
            complexity_entropy(a, return_dict=False) for a in attributions
        ]
        if return_dict:
            return {"complexity_score": complexity_score}
        return complexity_score

    with torch.no_grad():
        if not isinstance(attributions, tuple):
            attributions = (attributions,)

        # get batch size
        bsz = attributions[0].shape[0]

        # flatten feature tuple tensors into a single tensor of shape [batch_size, num_features]
        attributions_base, _ = _tuple_tensors_to_tensors(attributions)

        # flatten the feature dims into a single dim
        attributions = (
            attributions_base.view(bsz, -1).float().abs()
        )  # add epsilon to avoid zero attributions

        # see if there is any case where the attribution is zero
        for attribution in attributions:
            if torch.norm(attribution) == 0:
                attribution += 1e-8

        # normalize the attributions to get fractional contribution
        attributions = attributions / torch.sum(attributions, dim=1, keepdim=True)

        # compute batch-wise entropy of the fractional contribution
        complexity_score = Categorical(probs=attributions).entropy()
        if return_dict:
            return {"complexity_score": complexity_score}
        return complexity_score


def complexity_entropy_feature_grouped(
    attributions: Union[Tuple[torch.Tensor, ...], List[Tuple[torch.Tensor, ...]]],
    feature_mask: Tuple[torch.Tensor, ...] = None,
    is_multi_target: bool = False,
    return_dict: bool = False,
) -> torch.Tensor:
    """
    Implementation of Complexity metric by Bhatt et al., 2020. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library.

    Complexity of attributions is defined as the entropy of the fractional contribution of feature x_i to the total
    magnitude of the attribution. A complex explanation is one that uses all features in its explanation to explain
    some decision. Even though such an explanation may be faithful to the model output, if the number of features is
    too large it may be too difficult for the user to understand the explanations, rendering it useless.

    References:
        1) Umang Bhatt et al.: "Evaluating and aggregating
        feature-based model explanations." IJCAI (2020): 3016-3022.

    Args:
        attributions (Tuple[Tensor,...]): A tuple of tensors representing attributions of separate inputs. Each
            tensor in the tuple has shape (batch_size, num_features).
        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class.
                Default is False.
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
        >>> complexity_scores = complexity(attribution)
    """
    if is_multi_target:
        isinstance(
            attributions, list
        ), "attributions must be a list of tensors or list of tuples of tensors"
        complexity_score = [
            complexity_entropy_feature_grouped(a, return_dict=False)
            for a in attributions
        ]
        if return_dict:
            return {"complexity_score": complexity_score}
        return complexity_score

    def eval_complexity_entropy_feature_grouped_single_sample(
        attributions_single_sample, feature_mask_single_sample
    ):
        with torch.no_grad():
            if not isinstance(attributions_single_sample, tuple):
                attributions_single_sample = (attributions_single_sample,)

            # get batch size
            bsz = attributions_single_sample[0].shape[0]
            assert bsz == 1, "Batch size must be 1 for feature grouped complexity"

            # flatten all feature masks in the input
            if feature_mask_single_sample is not None:
                feature_mask_flattened, _ = _tuple_tensors_to_tensors(
                    feature_mask_single_sample
                )
            else:
                feature_mask_single_sample = _construct_default_feature_mask(
                    attributions_single_sample
                )
                feature_mask_flattened, _ = _tuple_tensors_to_tensors(
                    feature_mask_single_sample
                )

            # flatten all attributions_single_sample in the input, this must be done after the feature masks are flattened as
            # feature masks may depened on attribution
            attributions_single_sample, _ = _tuple_tensors_to_tensors(
                attributions_single_sample
            )

            # validate feature masks are increasing non-negative
            _validate_feature_mask(feature_mask_flattened)

            # gather attribution scores of feature groups
            # this can be useful for efficiently summing up attributions of feature groups
            # this is why we need a single batch size as gathered attributes and number of features for each
            # sample can be different
            reduced_attributions, n_features = _reduce_tensor_with_indices(
                attributions_single_sample[0],
                indices=feature_mask_flattened[0].flatten(),
            )
            reduced_attributions = reduced_attributions.abs()

            # see if there is any case where the attribution is zero
            if torch.norm(reduced_attributions) == 0:
                reduced_attributions += 1e-8
            # normalize the attributions to get fractional contribution
            reduced_attributions = reduced_attributions / torch.sum(
                reduced_attributions
            )
            return Categorical(probs=reduced_attributions).entropy()

    bsz = attributions[0].size(0)
    complexity_entropy_batch = []
    for sample_idx in range(bsz):
        complexity_entropy_score = (
            eval_complexity_entropy_feature_grouped_single_sample(
                attributions_single_sample=tuple(
                    attr[sample_idx].unsqueeze(0) for attr in attributions
                ),
                feature_mask_single_sample=(
                    tuple(mask[sample_idx].unsqueeze(0) for mask in feature_mask)
                    if feature_mask is not None
                    else None
                ),
            )
        )
        complexity_entropy_batch.append(complexity_entropy_score)
    complexity_entropy_batch = torch.tensor(complexity_entropy_batch)
    if return_dict:
        return {"complexity_score": complexity_entropy_batch}
    return complexity_entropy_batch
