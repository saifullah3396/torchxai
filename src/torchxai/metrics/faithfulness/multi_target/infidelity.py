#!/usr/bin/env python3

from typing import Any, Callable, List, Tuple, Union, cast

import torch
from captum._utils.common import (
    ExpansionTypes,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    safe_div,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.metrics._utils.batching import _divide_and_aggregate_metrics
from torch import Tensor

from torchxai.explainers._utils import _run_forward_multi_target


def _multi_target_infidelity(
    forward_func: Callable,
    perturb_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions_list: List[TensorOrTupleOfTensorsGeneric],
    baselines: BaselineType = None,
    additional_forward_args: Any = None,
    targets_list: List[TargetType] = None,
    feature_masks: TensorOrTupleOfTensorsGeneric = None,
    frozen_features: List[int] = None,
    n_perturb_samples: int = 10,
    max_examples_per_batch: int = None,
    normalize: bool = False,
) -> List[Tensor]:

    def _generate_perturbations(
        current_n_perturb_samples: int,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
        r"""
        The perturbations are generated for each example
        `current_n_perturb_samples` times.

        For performance reasons we are not calling `perturb_func` on each example but
        on a batch that contains `current_n_perturb_samples`
        repeated instances per example.
        """

        def call_perturb_func():
            r""" """
            baselines_pert = None
            inputs_pert: Union[Tensor, Tuple[Tensor, ...]]
            if len(inputs_expanded) == 1:
                inputs_pert = inputs_expanded[0]
                if baselines_expanded is not None:
                    baselines_pert = cast(Tuple, baselines_expanded)[0]
            else:
                inputs_pert = inputs_expanded
                baselines_pert = baselines_expanded
            return (
                perturb_func(
                    inputs=inputs_pert,
                    feature_masks=feature_masks_expanded,
                    frozen_features=frozen_features,
                    baselines=baselines_pert,
                )
                if baselines_pert is not None
                else perturb_func(
                    inputs=inputs_pert,
                    feature_masks=feature_masks_expanded,
                    frozen_features=frozen_features,
                )
            )

        inputs_expanded = tuple(
            torch.repeat_interleave(input, current_n_perturb_samples, dim=0)
            for input in inputs
        )
        feature_masks_expanded = None
        if feature_masks is not None:
            feature_masks_expanded = tuple(
                torch.repeat_interleave(feature_mask, current_n_perturb_samples, dim=0)
                for feature_mask in feature_masks
            )

        baselines_expanded = baselines
        if baselines is not None:
            baselines_expanded = tuple(
                (
                    baseline.repeat_interleave(current_n_perturb_samples, dim=0)
                    if isinstance(baseline, torch.Tensor)
                    and baseline.shape[0] == input.shape[0]
                    and baseline.shape[0] > 1
                    else baseline
                )
                for input, baseline in zip(inputs, cast(Tuple, baselines))
            )

        return call_perturb_func()

    def _validate_inputs_and_perturbations(
        inputs: Tuple[Tensor, ...],
        inputs_perturbed: Tuple[Tensor, ...],
        perturbations: Tuple[Tensor, ...],
    ) -> None:
        # asserts the sizes of the perturbations and inputs
        assert len(perturbations) == len(inputs), (
            """The number of perturbed
            inputs and corresponding perturbations must have the same number of
            elements. Found number of inputs is: {} and perturbations:
            {}"""
        ).format(len(perturbations), len(inputs))

        # asserts the shapes of the perturbations and perturbed inputs
        for perturb, input_perturbed in zip(perturbations, inputs_perturbed):
            assert perturb[0].shape == input_perturbed[0].shape, (
                """Perturbed input
                and corresponding perturbation must have the same shape and
                dimensionality. Found perturbation shape is: {} and the input shape
                is: {}"""
            ).format(perturb[0].shape, input_perturbed[0].shape)

    def _next_infidelity_tensors(
        current_n_perturb_samples: int,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]:
        perturbations, inputs_perturbed = _generate_perturbations(
            current_n_perturb_samples
        )

        perturbations = _format_tensor_into_tuples(perturbations)
        inputs_perturbed = _format_tensor_into_tuples(inputs_perturbed)

        _validate_inputs_and_perturbations(
            cast(Tuple[Tensor, ...], inputs),
            cast(Tuple[Tensor, ...], inputs_perturbed),
            cast(Tuple[Tensor, ...], perturbations),
        )

        targets_expanded_list = [
            _expand_target(
                target,
                current_n_perturb_samples,
                expansion_type=ExpansionTypes.repeat_interleave,
            )
            for target in targets_list
        ]
        additional_forward_args_expanded = _expand_additional_forward_args(
            additional_forward_args,
            current_n_perturb_samples,
            expansion_type=ExpansionTypes.repeat_interleave,
        )

        inputs_perturbed_fwd = _run_forward_multi_target(
            forward_func,
            inputs_perturbed,
            targets_expanded_list,
            additional_forward_args_expanded,
        )
        inputs_fwd = _run_forward_multi_target(
            forward_func, inputs, targets_list, additional_forward_args
        )
        inputs_fwd = torch.repeat_interleave(
            inputs_fwd, current_n_perturb_samples, dim=0
        )
        perturbed_fwd_diffs = inputs_fwd - inputs_perturbed_fwd
        perturbed_fwd_diffs_list = [
            perturbed_fwd_diffs[:, i] for i in range(perturbed_fwd_diffs.shape[1])
        ]
        # reshape as Tensor(bsz, current_n_perturb_samples)
        perturbed_fwd_diffs_list = [
            perturbed_fwd_diffs.view(bsz, -1)
            for perturbed_fwd_diffs in perturbed_fwd_diffs_list
        ]

        def attr_times_perturb(attributions, perturbations):
            attributions_expanded = tuple(
                torch.repeat_interleave(attribution, current_n_perturb_samples, dim=0)
                for attribution in attributions
            )

            attributions_times_perturb = tuple(
                (attribution_expanded * perturbation).view(
                    attribution_expanded.size(0), -1
                )
                for attribution_expanded, perturbation in zip(
                    attributions_expanded, perturbations
                )
            )

            attr_times_perturb_sums = sum(
                torch.sum(attribution_times_perturb, dim=1)
                for attribution_times_perturb in attributions_times_perturb
            )
            attr_times_perturb_sums = cast(Tensor, attr_times_perturb_sums)
            return attr_times_perturb_sums.view(bsz, -1)

        attr_times_perturb_list = [
            attr_times_perturb(attributions, perturbations)
            for attributions in attributions_list
        ]

        if normalize:
            # in order to normalize, we have to aggregate the following tensors
            # to calculate MSE in its polynomial expansion:
            # (a-b)^2 = a^2 - 2ab + b^2
            return [
                (
                    attr_times_perturb_sums.pow(2).sum(-1),
                    (attr_times_perturb_sums * perturbed_fwd_diffs).sum(-1),
                    perturbed_fwd_diffs.pow(2).sum(-1),
                )
                for perturbed_fwd_diffs, attr_times_perturb_sums in zip(
                    perturbed_fwd_diffs_list, attr_times_perturb_list
                )
            ]
        else:
            # returns (a-b)^2 if no need to normalize
            return [
                ((attr_times_perturb_sums - perturbed_fwd_diffs).pow(2).sum(-1),)
                for perturbed_fwd_diffs, attr_times_perturb_sums in zip(
                    perturbed_fwd_diffs_list, attr_times_perturb_list
                )
            ]

    def _sum_infidelity_tensor_lists(agg_tensors_list, tensors_list):
        return [
            tuple(agg_t + t for agg_t, t in zip(agg_tensors, tensors))
            for agg_tensors, tensors in zip(agg_tensors_list, tensors_list)
        ]

    isinstance(
        attributions_list, list
    ), "attributions must be a list of tensors or list of tuples of tensors"
    assert isinstance(targets_list, list), "targets must be a list of targets"
    assert all(
        isinstance(x, int) for x in targets_list
    ), "targets must be a list of ints"
    assert len(targets_list) == len(attributions_list), (
        """The number of targets in the targets_list and
        attributions_list must match. Found number of targets in the targets_list is: {} and in the
        attributions_list: {}"""
    ).format(len(targets_list), len(attributions_list))

    # perform argument formattings
    inputs = _format_tensor_into_tuples(inputs)  # type: ignore
    if baselines is not None:
        baselines = _format_baseline(baselines, cast(Tuple[Tensor, ...], inputs))
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    attributions_list = [_format_tensor_into_tuples(attributions) for attributions in attributions_list]  # type: ignore

    # Make sure that inputs and corresponding attributions have matching sizes.
    assert len(inputs) == len(attributions_list[0]), (
        """The number of tensors in the inputs and
        attributions must match. Found number of tensors in the inputs is: {} and in the
        attributions: {}"""
    ).format(len(inputs), len(attributions_list[0]))
    for inp, attr in zip(inputs, attributions_list[0]):
        assert inp.shape == attr.shape, (
            """Inputs and attributions must have
        matching shapes. One of the input tensor's shape is {} and the
        attribution tensor's shape is: {}"""
        ).format(inp.shape, attr.shape)

    bsz = inputs[0].size(0)
    with torch.no_grad():
        # if not normalize, directly return aggrgated MSE ((a-b)^2,)
        # else return aggregated MSE's polynomial expansion tensors (a^2, ab, b^2)
        agg_tensors = _divide_and_aggregate_metrics(
            cast(Tuple[Tensor, ...], inputs),
            n_perturb_samples,
            _next_infidelity_tensors,
            agg_func=_sum_infidelity_tensor_lists,
            max_examples_per_batch=max_examples_per_batch,
        )

    if normalize:

        def normalize_infidelities(agg_tensors):
            beta_num = agg_tensors[1]
            beta_denorm = agg_tensors[0]

            beta = safe_div(beta_num, beta_denorm)

            infidelity_values = (
                beta**2 * agg_tensors[0] - 2 * beta * agg_tensors[1] + agg_tensors[2]
            )
            return infidelity_values

        infidelity_values = [normalize_infidelities(x) for x in agg_tensors]
    else:
        infidelity_values = [x[0] for x in agg_tensors]

    infidelity_values = [x / n_perturb_samples for x in infidelity_values]

    return infidelity_values
