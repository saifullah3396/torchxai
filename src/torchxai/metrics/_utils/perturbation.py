#!/usr/bin/env python3

from typing import Tuple

import numpy as np
import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from dacite import Optional


def default_zero_baseline_func():
    def wrapped(
        inputs: TensorOrTupleOfTensorsGeneric,
        perturbation_masks: TensorOrTupleOfTensorsGeneric,
        baselines: Optional[TensorOrTupleOfTensorsGeneric] = None,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if not isinstance(perturbation_masks, tuple):
            perturbation_masks = (perturbation_masks,)
        zero_baselines = tuple(
            torch.zeros_like(x, device=x.device).float() for x in inputs
        )
        for input, mask, noisy_baseline in zip(
            inputs, perturbation_masks, zero_baselines
        ):
            input[mask.expand_as(input)] = noisy_baseline[mask.expand_as(input)]
        return inputs

    return wrapped


def default_perturb_func(noise_scale: float = 0.02):
    def wrapped(
        inputs: TensorOrTupleOfTensorsGeneric,
        perturbation_masks: TensorOrTupleOfTensorsGeneric,
        baselines: Optional[TensorOrTupleOfTensorsGeneric] = None,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if not isinstance(perturbation_masks, tuple):
            perturbation_masks = (perturbation_masks,)
        if baselines is not None and not isinstance(baselines, tuple):
            baselines = (baselines,)
        if baselines is None:
            # generate random noise if baselines are not provided
            baselines = tuple(
                torch.tensor(
                    np.random.uniform(low=-noise_scale, high=noise_scale, size=x.shape),
                    device=x.device,
                ).float()
                for x in inputs
            )
        for input, mask, baseline in zip(inputs, perturbation_masks, baselines):
            input[mask.expand_as(input)] = baseline[mask.expand_as(input)]
        return inputs

    return wrapped
