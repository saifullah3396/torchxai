#!/usr/bin/env python3


import numpy as np
import torch


def selectivity(descending_perturbation_fwds: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        from scipy.integrate import simps
        return simps(
            descending_perturbation_fwds,
            np.arange(0, descending_perturbation_fwds.shape[0]) / (descending_perturbation_fwds.shape[0] - 1),
        )
