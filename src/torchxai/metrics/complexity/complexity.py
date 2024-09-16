from logging import warning
from typing import Tuple

import torch
from torch.distributions import Categorical

from torchxai.metrics._utils.common import _tuple_tensors_to_tensors


def complexity(attributions: Tuple[torch.Tensor, ...]):
    """
    Implementation of Complexity metric by Bhatt et al., 2020.

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

        # see if there is any case where athe attribution is zero
        for attribution in attributions:
            if torch.all(attribution == 0):
                attribution += 1.0e-8

        # compute batch-wise fractional contribution of each feature
        attributions_abs = attributions.abs()
        total_sum = torch.sum(attributions_abs, dim=1)
        fractional_contribution = attributions_abs / (
            (total_sum.unsqueeze(-1).expand(attributions_abs.shape))
        )

        # compute batch-wise entropy of the fractional contribution
        return Categorical(probs=fractional_contribution).entropy()
