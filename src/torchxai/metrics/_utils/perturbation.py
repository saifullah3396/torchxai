#!/usr/bin/env python3

from typing import List, Tuple

import numpy as np
import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from dacite import Optional

from torchxai.metrics._utils.common import (
    _split_tensors_to_tuple_tensors,
    _tuple_tensors_to_tensors,
)


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
        assert perturbation_masks[0].dtype == torch.bool
        perturbation_masks = tuple(
            mask.expand_as(input) for mask, input in zip(perturbation_masks, inputs)
        )
        return tuple(~mask * input for input, mask in zip(inputs, perturbation_masks))

    return wrapped


def default_fixed_baseline_perturb_func():
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
        assert perturbation_masks[0].dtype == torch.bool

        perturbation_masks = tuple(
            mask.expand_as(input) for mask, input in zip(perturbation_masks, inputs)
        )
        return tuple(
            (~mask * input + mask * baseline)
            for input, mask, baseline in zip(inputs, perturbation_masks, baselines)
        )

    return wrapped


def default_random_perturb_func(noise_scale: float = 0.02):
    def wrapped(
        inputs: TensorOrTupleOfTensorsGeneric,
        perturbation_masks: TensorOrTupleOfTensorsGeneric,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if not isinstance(perturbation_masks, tuple):
            perturbation_masks = (perturbation_masks,)
        assert perturbation_masks[0].dtype == torch.bool

        # generate random noise if baselines are not provided
        random_baselines = tuple(
            torch.tensor(
                np.random.uniform(low=-noise_scale, high=noise_scale, size=x.shape),
                device=x.device,
            ).float()
            for x in inputs
        )
        perturbation_masks = tuple(
            mask.expand_as(input) for mask, input in zip(perturbation_masks, inputs)
        )
        perturbed_inputs = tuple(
            (~mask * input + mask * random_baseline)
            for input, mask, random_baseline in zip(
                inputs, perturbation_masks, random_baselines
            )
        )
        return perturbed_inputs

    return wrapped


def default_infidelity_perturb_fn(noise_scale: float = 0.003):
    def wrapped(
        inputs,
        baselines=None,
        feature_masks=None,
        frozen_features=None,
    ):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        noise = tuple(torch.randn_like(x) * noise_scale for x in inputs)
        if frozen_features is not None and feature_masks is not None:
            for n_batch, feature_mask_batch in zip(
                noise, feature_masks
            ):  # for each feature type
                for n_sample, feature_mask_sample, frozen_features_sample in zip(
                    n_batch,
                    feature_mask_batch,
                    frozen_features,  # for each sample in batch
                ):
                    for (
                        feature_idx
                    ) in frozen_features_sample:  # for each frozen feature
                        n_sample[feature_mask_sample == feature_idx] = 0
        return noise, tuple(x - n for x, n in zip(inputs, noise))

    return wrapped


def _feature_mask_to_perturbation_mask_n_indices(
    mask, feature_indices, frozen_features
):
    if frozen_features is not None:
        valid_indices_mask = ~torch.isin(feature_indices, frozen_features)
        feature_indices = feature_indices[valid_indices_mask]

    # create the perturbation mask in one operation
    perturbation_mask = torch.any(
        mask.unsqueeze(0) == feature_indices.unsqueeze(1), dim=0, keepdim=True
    )
    return perturbation_mask  # Shape: (num_features, mask.size)


def _generate_random_perturbation_masks(
    n_perturbations_per_sample: int,
    feature_mask: Tuple[torch.Tensor, ...],
    percent_features_perturbed=0.1,
    frozen_features: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, ...]:
    # Start of main logic
    flattened_feature_masks, flattened_feature_mask_base_shapes = (
        _tuple_tensors_to_tensors(feature_mask)
    )

    if frozen_features is not None:
        assert flattened_feature_masks.shape[0] == len(frozen_features), (
            "Number of samples in frozen_features should be equal to the number of samples in the feature_mask. "
            f"Got {flattened_feature_masks.shape[0]} samples in feature_mask and {len(frozen_features)} samples in frozen_features."
        )

    perturbation_masks = []
    for sample_idx, flattened_feature_mask in enumerate(flattened_feature_masks):
        flattened_feature_mask = flattened_feature_mask.squeeze()
        feature_indices_all = torch.unique(flattened_feature_mask)

        # compute total features to perturb
        total_features_perturbed = int(
            percent_features_perturbed * len(feature_indices_all)
        )

        # generate all random indices for perturbations in one go
        def generate_rand_indices():
            rand_feature_indices = feature_indices_all[
                torch.randperm(
                    len(feature_indices_all), device=flattened_feature_mask.device
                )
            ]
            if frozen_features is not None:
                valid_mask = ~torch.isin(
                    rand_feature_indices, frozen_features[sample_idx]
                )
                rand_feature_indices = rand_feature_indices[valid_mask]

            return rand_feature_indices[:total_features_perturbed].unsqueeze(0)

        rand_indices = torch.cat(
            [generate_rand_indices() for _ in range(n_perturbations_per_sample)],
            dim=0,
        )

        # generate all perturbation masks at once
        perturbation_masks_per_sample = torch.cat(
            [
                (
                    flattened_feature_mask.unsqueeze(0)
                    == curr_rand_indices.unsqueeze(1)
                ).any(dim=0, keepdim=True)
                for curr_rand_indices in rand_indices
            ]
        )
        perturbation_masks.append(perturbation_masks_per_sample)
    perturbation_masks = torch.cat(perturbation_masks, dim=0)
    perturbation_masks = _split_tensors_to_tuple_tensors(
        perturbation_masks, flattened_feature_mask_base_shapes
    )
    bsz = flattened_feature_masks.shape[0]
    perturbation_masks = tuple(
        x.view(bsz, n_perturbations_per_sample, *x.shape[1:])
        for x in perturbation_masks
    )
    return perturbation_masks


def perturb_fn_drop_batched_single_output(
    feature_mask: Tuple[torch.Tensor, ...],
    percent_features_perturbed=0.1,
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
            n_perturbations_per_sample=current_batch_size // total_samples,
            feature_mask=feature_mask,
            percent_features_perturbed=percent_features_perturbed,
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
