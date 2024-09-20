import torch


def generate_mask_weights(feature_mask_batch):
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
