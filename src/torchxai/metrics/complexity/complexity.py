#!/usr/bin/env python3

from typing import Tuple

import torch
from torch.distributions import Categorical

from torchxai.metrics._utils.common import _tuple_tensors_to_tensors


def complexity(
    attributions: Tuple[torch.Tensor, ...], is_multi_target: bool = False
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
        return [complexity(a) for a in attributions]

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
        return Categorical(probs=attributions).entropy()
