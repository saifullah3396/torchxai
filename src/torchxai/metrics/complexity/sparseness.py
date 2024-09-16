from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from torchxai.metrics._utils.common import _tuple_tensors_to_tensors


def sparseness(attributions: Tuple[Tensor, ...]):
    """
    Implementation of Sparseness metric by Chalasani et al., 2020.

    Sparseness is quantified using the Gini Index applied to the vector of the absolute values of attributions. The
    test asks that features that are truly predictive of the output F(x) should have significant contributions, and
    similarly, that irrelevant (or weakly-relevant) features should have negligible contributions.

    Assumptions:
        - Based on the implementation of the authors as found on the following link:
        <https://github.com/jfc43/advex/blob/master/DNN-Experiments/Fashion-MNIST/utils.py>.

    Args:
        attributions (Tuple[Tensor,...]): A tuple of tensors representing attributions of separate inputs. Each
            tensor in the tuple has shape (batch_size, num_features).
        eps (float): The threshold value for attributions to be considered important.
    Returns:
        Tensor: The complexity of each attribution in the batch.
    """
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
        return torch.tensor(
            [gini_index(attribution.numpy() + 1e-8) for attribution in attributions]
        )
