#!/usr/bin/env python3

from typing import Tuple

import numpy as np
import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from dacite import Optional
from ignite.utils import convert_tensor


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


def default_random_perturb_func(noise_scale: float = 0.02):
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
            if len(input.shape) != len(mask.shape):
                mask = mask.unsqueeze(-1)

            input[mask.expand_as(input)] = baseline[mask.expand_as(input)]
        return inputs

    return wrapped


def _generate_random_perturbation_masks(
    total_perturbations_per_feature_group: int,
    feature_mask: Tuple[torch.Tensor, ...],
    perturbation_probability: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, ...]:
    if not isinstance(feature_mask, tuple):
        feature_mask = (feature_mask,)

    # convert to device
    feature_mask = convert_tensor(feature_mask, device=device)

    perturbation_masks = tuple()
    for feature_mask_per_type in feature_mask:  # unpack the tuples of feature types
        perturbations_per_input_type = []
        for (
            feature_mask_per_sample
        ) in feature_mask_per_type:  # unpack the samples in a batch
            features_in_sample = torch.unique(feature_mask_per_sample)
            perturbation_masks_per_sample = torch.zeros(
                (
                    total_perturbations_per_feature_group,
                    *feature_mask_per_sample.shape,
                ),
                dtype=torch.bool,
                device=device,
            )
            for i in range(total_perturbations_per_feature_group):
                feature_drop_mask = (
                    torch.rand(len(features_in_sample), device=device)
                    < perturbation_probability
                )
                dropped_feature_indices = features_in_sample[feature_drop_mask]
                for dropped_feature_idx in dropped_feature_indices:
                    perturbation_masks_per_sample[i][
                        feature_mask_per_sample == dropped_feature_idx
                    ] = True
            perturbations_per_input_type.append(perturbation_masks_per_sample)
        perturbation_masks += (torch.cat(perturbations_per_input_type, dim=0),)
    return perturbation_masks


def perturb_fn_drop_batched_single_output(
    feature_mask: Tuple[torch.Tensor, ...],
    drop_probability=0.1,
):
    def wrapped(inputs, baselines):
        # to compute infidelity we take randomly set half the features to baseline
        # here we generate random indices which will be set to baseline
        # input shape should be (batch_size, seq_length, feature_dim)
        # first we generate rand boolean vectors of size (batch_size, seq_length)
        # then we repeat each bool value n times where n is the number of features in the group given by "repeats"
        # then the input expanded to feature_dim
        # Note: This happens even in cases where features are not grouped together because we want the token
        # removal frequency to be the same among all attribution methods whether it uses grouped features or not
        # for example for deep_lift method the feature groups=n_tokens + 1 (CLS) + 1 (SEP) + n_pad_tokens (PAD)
        # but for lime method the feature groups=n_words (each word consists of m tokens) + 1 (CLS) + 1 (SEP) + n_pad_tokens (PAD)

        if not isinstance(inputs, tuple):
            inputs = (inputs,)
            baselines = (baselines,)

        total_samples = feature_mask[0].shape[0]
        current_batch_size = inputs[0].shape[0]
        perturbation_masks = _generate_random_perturbation_masks(
            total_perturbations_per_feature_group=current_batch_size // total_samples,
            feature_mask=feature_mask,
            perturbation_probability=drop_probability,
            device=inputs[0].device,
        )

        # expand the perturbation masks to the input shape
        perturbation_masks = tuple(
            perturbation_mask.expand_as(input)
            for input, perturbation_mask in zip(inputs, perturbation_masks)
        )

        # create a copy of the input tensor
        inputs_perturbed = tuple(input.clone() for input in inputs)
        for input_perturbed, baseline, perturbation_mask in zip(
            inputs_perturbed, baselines, perturbation_masks
        ):
            input_perturbed[perturbation_mask] = baseline[perturbation_mask]

        # convert the perturbation masks to float
        perturbation_masks = tuple(x.float() for x in perturbation_masks)

        # if debugging:
        #     for x in inputs:
        #         from torchvision.utils import make_grid

        #         x = make_grid(x)
        #         import matplotlib.pyplot as plt

        #         plt.imshow(x.permute(1, 2, 0).cpu())
        #         plt.show()

        #     for x in inputs_perturbed:
        #         from torchvision.utils import make_grid

        #         x = make_grid(x)
        #         import matplotlib.pyplot as plt

        #         plt.imshow(x.permute(1, 2, 0).cpu())
        #         plt.show()

        # if debugging:
        #     import matplotlib.pyplot as plt

        #     for perturbation_mask, ptb in zip(perturbation_masks, inputs_perturbed):
        #         fig, axes = plt.subplots(figsize=(50, 10), nrows=2)
        #         axes[0].matshow(perturbation_mask[:, :, 0][:, :50].cpu().numpy())
        #         axes[1].matshow(ptb[:, :, 0][:, :50].cpu().numpy())
        #         plt.show()

        if len(perturbation_masks) == 1:
            perturbation_masks = perturbation_masks[0]
            inputs_perturbed = inputs_perturbed[0]

        return perturbation_masks, inputs_perturbed

    return wrapped
