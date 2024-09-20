import numpy as np
import torch


class GridSegmenter:
    def __init__(self, cell_size: int):
        self._cell_size = cell_size

    def __call__(self, arr: np.ndarray):
        h, w, _ = arr.shape
        feature_mask = np.arange(h // self._cell_size * w // self._cell_size).reshape(
            h // self._cell_size, w // self._cell_size
        )
        return np.kron(
            feature_mask, np.ones((self._cell_size, self._cell_size))
        ).astype(int)


def generate_mask_weights(feature_masks_batch):
    feature_masks_weighted_batch = torch.zeros_like(
        feature_masks_batch, dtype=torch.float
    )
    for feature_mask, feature_mask_weighted in zip(
        feature_masks_batch, feature_masks_weighted_batch
    ):  # batch iteration
        labels, counts = torch.unique(feature_mask, return_counts=True)
        for idx in range(labels.shape[0]):
            feature_mask_weighted[feature_mask == labels[idx]] = 1.0 / counts[idx]
    return feature_masks_weighted_batch
