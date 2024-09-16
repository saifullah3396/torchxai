#!/usr/bin/env python3

from typing import Tuple

import torch
from torchxai.metrics._utils.common import _tuple_tensors_to_tensors


def effective_complexity(
    attributions: Tuple[torch.Tensor, ...],
    eps: float = 1.0e-5,
    normalize_attribution: bool = True,
) -> torch.Tensor:
    """
    Implementation of Effective complexity metric by Nguyen at el., 2020. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library.

    Effective complexity measures how many attributions in absolute values are exceeding a certain threshold (eps)
    where a value above the specified threshold implies that the features are important and under indicates it is not.
    Effective complexity requires the attributions to be normalized to return reasonable outputs since the original
    attributions may have different scales and effective complexity is sensitive to the scale of the attributions.

    References:
        1) An-phi Nguyen and María Rodríguez Martínez.: "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).

    Args:
        attributions (Tuple[Tensor,...]): A tuple of tensors representing attributions of separate inputs. Each
            tensor in the tuple has shape (batch_size, num_features).
        eps (float): The threshold value for attributions to be considered important.
        normalize_attribution (bool): If True, the attributions are normalized to sum to 1.
    Returns:
        Tensor: A tensor of scalar effective complexity per
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
        >>> effective_complexity_scores = effective_complexity(attribution)
    """

    with torch.no_grad():
        if not isinstance(attributions, tuple):
            attributions = (attributions,)

        # get batch size
        bsz = attributions[0].shape[0]

        # flatten feature tuple tensors into a single tensor of shape [batch_size, num_features]
        attributions, _ = _tuple_tensors_to_tensors(attributions)

        # flatten the feature dims into a single dim
        attributions = attributions.view(bsz, -1).float()

        # normalize the attributions sample-wise in the batch if required
        if normalize_attribution:
            attributions = attributions / torch.sum(attributions, dim=1, keepdim=True)

        # compute batch-wise effective complexity of the attribution map
        return torch.sum(attributions.abs() > eps, dim=1)
