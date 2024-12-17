import numpy as np
import torch


def compute_abpc(aopc_scores: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        morf = (
            aopc_scores[:, 0, :].numpy().mean(0)
        )  # first row is descending, take mean over the dataset
        lerf = (
            aopc_scores[:, 1, :].numpy().mean(0)
        )  # second row is ascending, take mean over the dataset
        abpc = morf - lerf
        return abpc


def compute_abpc_scores(
    morf_scores: torch.Tensor, lerf_scores: torch.Tensor
) -> torch.Tensor:
    return compute_abpc(
        torch.stack(
            [torch.from_numpy(morf_scores), torch.from_numpy(lerf_scores)], dim=1
        )
    )
