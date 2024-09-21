
import torch


def _generate_mask_weights(feature_mask_batch: torch.Tensor) -> torch.Tensor:
    """
    This function takes a batch of feature masks and generates a corresponding
    batch of weighted feature masks. Each unique feature in the mask is assigned
    a weight that is inversely proportional to its frequency in the mask.
    Args:
        feature_mask_batch (torch.Tensor): A batch of feature masks with shape (batch_size, ...), where each element
            is an integer representing a feature.
    Returns:
        torch.Tensor: A batch of weighted feature masks with the same shape as `feature_mask_batch`, where each
            feature is weighted by the inverse of its frequency in the mask.
    """

    feature_mask_weighted_batch = torch.zeros_like(
        feature_mask_batch, dtype=torch.float
    )
    for feature_mask, feature_mask_weighted in zip(
        feature_mask_batch, feature_mask_weighted_batch
    ):  # batch iteration
        labels, counts = torch.unique(feature_mask, return_counts=True)
        for idx in range(labels.shape[0]):
            feature_mask_weighted[feature_mask == labels[idx]] = 1.0 / counts[idx]
    return feature_mask_weighted_batch
