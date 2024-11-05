#!/usr/bin/env python3

from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchxai.metrics._utils.common import _tuple_tensors_to_tensors


def sparseness(
    attributions: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]],
    is_multi_target: bool = False,
    return_dict: bool = False,
) -> Tensor:
    """
    Implementation of Sparseness metric by Chalasani et al., 2020. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library.

    Sparseness is quantified using the Gini Index applied to the vector of the absolute values of attributions. The
    test asks that features that are truly predictive of the output F(x) should have significant contributions, and
    similarly, that irrelevant (or weakly-relevant) features should have negligible contributions. A higher sparseness
    score indicates that the attributions are more sparse, i.e., a few features have high attribution values and the
    rest have low attribution values. This is desirable as it indicates that the model is using a few features to make
    decisions.

    Sparseness does not require the attributions be normalized to return correct outputs.

    Assumptions:
        - Based on the implementation of the authors as found on the following link:
        <https://github.com/jfc43/advex/blob/master/DNN-Experiments/Fashion-MNIST/utils.py>.

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
        Tensor: A tensor of scalar sparseness scores per
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
        >>> sparseness_scores = sparseness(attribution)
    """
    if is_multi_target:
        isinstance(
            attributions, list
        ), "attributions must be a list of tensors or list of tuples of tensors"
        sparseness_score = [sparseness(a, return_dict=False) for a in attributions]
        if return_dict:
            return {"sparseness_score": sparseness_score}
        return sparseness_score

    with torch.no_grad():
        if not isinstance(attributions, tuple):
            attributions = (attributions,)

        # get batch size
        bsz = attributions[0].shape[0]

        # flatten feature tuple tensors into a single tensor of shape [batch_size, num_features]
        attributions, _ = _tuple_tensors_to_tensors(attributions)

        # flatten the feature dims into a single dim, take absolute values and sort them batch-wise
        attributions = attributions.view(bsz, -1).float().abs().sort(dim=1).values

        def gini_index(vec: np.ndarray):
            return (
                np.sum((2 * np.arange(1, vec.shape[0] + 1) - vec.shape[0] - 1) * vec)
            ) / (vec.shape[0] * np.sum(vec))

        # compute batch-wise Gini Index of the attribution map
        sparseness_score = torch.tensor(
            [
                gini_index(attribution.detach().cpu().numpy() + 1e-8)
                for attribution in attributions
            ],
            device=attributions.device,
        ).float()
        if return_dict:
            return {"sparseness_score": sparseness_score}
        return sparseness_score
