#!/usr/bin/env python3


import numpy as np
import torch


def compute_abpc_scores(aopc_scores: torch.Tensor) -> torch.Tensor:
    """
    Computes the ABPC score for the given AOPC scores over the complete dataset.

    Args:
        aopc_scores: torch.Tensor: The AOPC scores for the complete dataset. The shape of the tensor is
            [n_samples, (2*n_random_perms), n_steps]. The first row for each sample corresponds to the descending
            order of feature importance, the second row corresponds to the ascending order of feature importance and
            the rest of the rows correspond to the random order of feature importance.
    """

    from scipy.integrate import simps

    n_steps = aopc_scores.shape[1]
    morf = np.array(aopc_scores[:, 0, :]).mean(
        0
    )  # first row is descending, take mean over the dataset
    lerf = np.array(aopc_scores[:, 1, :]).mean(
        0
    )  # second row is ascending, take mean over the dataset

    morf_mean = simps(
        morf, np.arange(0, n_steps)
    )  # find the area under the curve for descending order
    lerf_mean = simps(
        lerf, np.arange(0, n_steps)
    )  # find the area under the curve for ascending order

    abpc_score = morf_mean - lerf_mean
    return abpc_score
