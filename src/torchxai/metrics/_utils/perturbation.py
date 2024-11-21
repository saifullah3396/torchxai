#!/usr/bin/env python3

from typing import Tuple, Union

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
            (~mask * input + mask * baseline)
            for input, mask, baseline in zip(
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


def _generate_random_perturbation_masks(
    n_perturbations_per_sample: int,
    feature_mask: Tuple[torch.Tensor, ...],
    perturbation_probability: float = 0.1,
    frozen_features: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, ...]:
    """
    Generate random perturbation masks for a given feature mask.
    """
    if not isinstance(feature_mask, tuple):
        feature_mask = (feature_mask,)

    # convert to device
    feature_mask = convert_tensor(feature_mask, device=device)

    perturbation_masks = tuple()
    for feature_mask_per_type in feature_mask:  # unpack the tuples of feature types
        perturbations_per_input_type = []
        for sample_idx, feature_mask_per_sample in enumerate(
            feature_mask_per_type
        ):  # unpack the samples in a batch
            # feature_mask_per_sample could be a tensor of shape (channel, height, width) or (seq_length, feature_dim)
            # find the total number of unique feature groups in the sample
            features_in_sample = torch.unique(feature_mask_per_sample)

            # generate an empty mask for this sample, this could be a tensor of shape same as
            # (n_perturbations_per_sample, channel, height, width) or (n_perturbations_per_sample, seq_length, feature_dim)
            # where sample shape is (channel, height, width) or (seq_length, feature_dim)
            # since we wish to generate n_perturbations_per_sample random perturabtions for each sample
            perturbation_masks_per_sample = torch.zeros(
                (
                    n_perturbations_per_sample,
                    *feature_mask_per_sample.shape,
                ),
                dtype=torch.bool,
                device=device,
            )
            for i in range(n_perturbations_per_sample):
                # here we generate a single random perturbation mask for the sample. This would be of shape
                # (channel, height, width) or (seq_length, feature_dim)
                # we randomly drop features with probability perturbation_probability
                # if total feature groups are features_in_sample this will essentially return a mask
                # with features_in_sample * perturbation_probability features dropped
                feature_drop_mask = (
                    torch.rand(len(features_in_sample), device=device)
                    < perturbation_probability
                )

                # freeze some features if required
                if frozen_features is not None:
                    for frozen_idx in frozen_features[sample_idx]:
                        if frozen_idx in features_in_sample:
                            feature_drop_mask[features_in_sample == frozen_idx] = False

                # here we take the indices of the feature groups that are dropped
                # this means if the feature mask is like [0, 0, 0, 1, 1, 1, 2, 2, 2]
                # the features_in_sample will be [0, 1, 2] and if the feature_drop_mask is [True, False, True]
                # the dropped_feature_indices will be [0, 2], meaning we need to drop 0 and 2 feature groups
                dropped_feature_indices = features_in_sample[feature_drop_mask]

                # here we set the perturbation mask to True where the feature group is dropped
                for dropped_feature_idx in dropped_feature_indices:
                    perturbation_masks_per_sample[i][
                        feature_mask_per_sample == dropped_feature_idx
                    ] = True

            # now we have n_perturbations_per_sample random perturbation masks for this sample so we append to list
            # for batch conversion later
            perturbations_per_input_type.append(perturbation_masks_per_sample)
        # now we have n_perturbations_per_sample random perturbation masks for all samples in the batch
        # we concatenate them to form a single tensor of shape (n_perturbations_per_sample * batch_size, ...)
        perturbation_masks += (torch.stack(perturbations_per_input_type, dim=0),)

    # output is a tuple of tensors each of shape
    # ((batch_size, n_perturbations_per_sample, *input_shape), ...)
    return perturbation_masks


def _generate_random_perturbation_masks_with_fixed_n(
    n_perturbations_per_sample: int,
    feature_mask: Tuple[torch.Tensor, ...],
    n_features_perturbed: Union[int, float] = 1,
    frozen_features: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, ...]:
    """
    Generate random perturbation masks for a given feature mask.
    """
    if not isinstance(feature_mask, tuple):
        feature_mask = (feature_mask,)

    # convert to device
    feature_mask = convert_tensor(feature_mask, device=device)

    # get batch size
    batch_size = feature_mask[0].shape[0]
    perturbation_masks = []
    for sample_idx in range(batch_size):
        feature_masks_per_sample = [x[sample_idx] for x in feature_mask]
        features_in_sample = torch.unique(
            torch.cat(tuple(x.flatten() for x in feature_masks_per_sample))
        )
        perturbation_masks_per_sample = []
        for _ in range(n_perturbations_per_sample):
            # here we generate a single random perturbation mask for the sample. This would be of shape
            # (channel, height, width) or (seq_length, feature_dim)
            # we randomly drop n_features_perturbed features
            if isinstance(n_features_perturbed, float):
                assert (
                    0 <= n_features_perturbed <= 1
                ), "n_features_perturbed should be in [0, 1] if passed as a float percentage of total features"
                n_features_perturbed = int(
                    n_features_perturbed * len(features_in_sample)
                )
            feature_drop_mask = torch.randperm(len(features_in_sample), device=device)[
                :n_features_perturbed
            ]

            # here we take the indices of the feature groups that are dropped
            # this means if the feature mask is like [0, 0, 0, 1, 1, 1, 2, 2, 2]
            # the features_in_sample will be [0, 1, 2] and if the feature_drop_mask is [True, False, True]
            # the dropped_feature_indices will be [0, 2], meaning we need to drop 0 and 2 feature groups
            dropped_feature_indices = features_in_sample[feature_drop_mask]

            # freeze some features if required
            if frozen_features is not None:
                for frozen_idx in frozen_features[sample_idx]:
                    if frozen_idx in dropped_feature_indices:
                        dropped_feature_indices = dropped_feature_indices[
                            dropped_feature_indices != frozen_idx
                        ]

            perturbation_masks_per_feature_type = tuple()
            for feature_masks_per_sample_per_type in feature_masks_per_sample:
                # generate an empty mask for this sample, this could be a tensor of shape same as
                # (n_perturbations_per_sample, channel, height, width) or (n_perturbations_per_sample, seq_length, feature_dim)
                # where sample shape is (channel, height, width) or (seq_length, feature_dim)
                # since we wish to generate n_perturbations_per_sample random perturabtions for each sample
                mask = torch.zeros_like(
                    feature_masks_per_sample_per_type,
                    dtype=torch.bool,
                    device=device,
                )
                # here we set the perturbation mask to True where the feature group is dropped
                for dropped_feature_idx in dropped_feature_indices:
                    mask[feature_masks_per_sample_per_type == dropped_feature_idx] = (
                        True
                    )
                perturbation_masks_per_feature_type += (mask,)
            perturbation_masks_per_sample.append(perturbation_masks_per_feature_type)
        perturbation_masks.append(perturbation_masks_per_sample)

    perturbation_masks_unwrapped = tuple(
        [
            [
                perturbation_masks_per_feature_type[feature_type_idx].tolist()
                for perturbation_masks_per_feature_type in perturbation_masks_per_sample
            ]
            for perturbation_masks_per_sample in perturbation_masks
        ]
        for feature_type_idx in range(len(feature_mask))
    )
    perturbation_masks_unwrapped = tuple(
        torch.tensor(x, device=device) for x in perturbation_masks_unwrapped
    )
    return perturbation_masks_unwrapped


def perturb_fn_drop_batched_single_output(
    feature_mask: Tuple[torch.Tensor, ...],
    perturbation_probability=0.1,
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
            perturbation_probability=perturbation_probability,
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
