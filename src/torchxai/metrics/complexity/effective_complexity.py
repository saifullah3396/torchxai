from typing import Tuple

import torch
from torch.distributions import Categorical

from torchxai.metrics._utils.common import _tuple_tensors_to_tensors


def effective_complexity(attributions: Tuple[torch.Tensor, ...], eps: float = 1.0e-5):
    """
    Implementation of Effective complexity metric by Nguyen at el., 2020.

    Effective complexity measures how many attributions in absolute values are exceeding a certain threshold (eps)
    where a value above the specified threshold implies that the features are important and under indicates it is not.

    References:
        1) An-phi Nguyen and María Rodríguez Martínez.: "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).

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

        # flatten the feature dims into a single dim
        attributions = attributions.view(bsz, -1).float()

        # compute batch-wise effective complexity of the attribution map
        return torch.sum(attributions.abs() > eps, dim=1)
