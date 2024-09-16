#!/usr/bin/env python3

import warnings
from calendar import c
from typing import Any, Callable, Tuple, Union, cast

import numpy as np
import scipy
import torch
from captum._utils.common import (
    ExpansionTypes,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    _run_forward,
    safe_div,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.log import log_usage
from captum.metrics._utils.batching import _divide_and_aggregate_metrics
from dacite import Optional
from torch import Tensor

from torchxai.metrics._utils.batching import _divide_and_aggregate_metrics_n_features
from torchxai.metrics._utils.common import (
    _construct_default_feature_masks,
    _reduce_tensor_with_indices,
    _split_tensors_to_tuple_tensors,
    _tuple_tensors_to_tensors,
    _validate_feature_mask,
)


@log_usage()
def eval_faithfulness_estimate_single_sample_tupled_computation(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType,
    feature_masks: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_examples_per_batch: int = None,
) -> Tensor:
    def _next_faithfulness_estimate_tensors(
        current_n_perturbed_features: int,
        current_n_steps: int,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]:
        # get the indices of features that will be perturbed in the current iteration
        # for example if we do 1 by 1 perturbation, then the first iteration will perturb the first feature
        # the second iteration will perturb both the first and second feature and so on
        inputs_expanded = tuple(
            inputs.repeat(
                current_n_perturbed_features, *tuple([1] * len(inputs.shape[1:]))
            )
        )
        baselines_expanded = tuple(
            baselines.repeat(
                current_n_perturbed_features, *tuple([1] * len(baselines.shape[1:]))
            )
        )
        perturbation_masks = tuple(
            torch.cat(
                [
                    mask == ascending_attribution_indices[: current_n_steps - idx]
                    for idx in range(current_n_perturbed_features, -1, 0)
                ]
            )
            for mask in feature_masks
        )

        baselines_perturbed = baselines_expanded
        for baseline, mask, input in zip(
            baselines_perturbed, perturbation_masks, inputs_expanded
        ):
            baseline[mask] = input[mask]

        targets_expanded = _expand_target(
            target,
            current_n_perturbed_features,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        additional_forward_args_expanded = _expand_additional_forward_args(
            additional_forward_args,
            current_n_perturbed_features,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        baselines_perturbed_fwd = _run_forward(
            forward_func,
            baselines_perturbed,
            targets_expanded,
            additional_forward_args_expanded,
        )
        baselines_perturbed_fwd = baselines_perturbed_fwd.chunk(
            current_n_perturbed_features
        )

        # this is a list of tensors which holds the forward outputs of the model
        # on each feature group perturbation
        # the first element will be when the feature with lowest importance is added to the baseline
        # the last element will be when all features are added to the baseline
        return (list(baselines_perturbed_fwd),)

    def _sum_faithfulness_estimate_tensors(agg_tensors, tensors):
        return tuple(agg_t + t for agg_t, t in zip(agg_tensors, tensors))

    bsz = inputs[0].size(0)
    assert bsz == 1, "Batch size must be 1 for faithfulness_estimate_single_sample"
    if feature_masks is not None:
        # assert that all elements in the feature_masks are unique and non-negative increasing
        _validate_feature_mask(feature_masks)
    else:
        feature_masks = _construct_default_feature_masks(attributions)

    # this assumes a batch size of 1, this will not work for batch size > 1
    n_features = max(x.max() for x in feature_masks).item() + 1

    # gather attribution scores of feature groups
    # this can be useful for efficiently summing up attributions of feature groups
    gathered_attributions = tuple()
    for attribution, feature_mask in zip(attributions, feature_masks):
        gathered_attribution = torch.zeros_like(attribution)
        reduced_indices = feature_mask.squeeze() - feature_mask.min()
        gathered_attribution.index_add_(1, reduced_indices, attribution.clone())
        gathered_attributions += (gathered_attribution[:, : reduced_indices.max() + 1],)

    total_features_in_attribution = sum(
        tuple(x.shape[1] for x in gathered_attributions)
    )
    assert total_features_in_attribution == n_features, (
        """The total number of features in the attribution scores
        must be equal to the total number of features found inside the feature masks in the inputs. Found
        total number of features in the attribution scores is: {} and in the
        inputs: {}"""
    ).format(total_features_in_attribution, n_features)

    # flatten the attributions to get the attribution scores of each feature group. Since this is for a single
    # sample, we can flatten the attributions to get the attribution scores of each feature group
    gathered_attributions = torch.cat(tuple(x.squeeze() for x in gathered_attributions))

    # get the gathererd-attributions sorted in ascending order of their importance
    ascending_attribution_indices = torch.argsort(gathered_attributions)
    with torch.no_grad():
        # the logic for this implementation as as follows:
        # we start from baseline and in each iteration, a feature group is replaced by the original sample
        # in ascending order of its importance
        agg_tensors = _divide_and_aggregate_metrics_n_features(
            n_features,
            _next_faithfulness_estimate_tensors,
            agg_func=_sum_faithfulness_estimate_tensors,
            max_examples_per_batch=max_examples_per_batch,
        )

        # compute faithfulness_estimate corr metric
        def compute_faithfulness_estimate(
            baseline_perturbed_fwds: np.ndarray,
        ):
            # as feature are added from least to higher importance, the forward outputs should be monotonically increasing
            return np.all(np.diff(baseline_perturbed_fwds) >= 0)

        agg_tensors = tuple(np.array(x) for x in agg_tensors)
        baseline_perturbed_fwds = agg_tensors[0]
        faithfulness_estimate = compute_faithfulness_estimate(baseline_perturbed_fwds)
    return faithfulness_estimate


@log_usage()
def faithfulness_estimate_tupled_computation(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType,
    feature_masks: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_examples_per_batch: int = None,
) -> Tensor:
    # perform argument formattings
    inputs = _format_tensor_into_tuples(inputs)  # type: ignore
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    attributions = _format_tensor_into_tuples(attributions)  # type: ignore

    # Make sure that inputs and corresponding attributions have matching sizes.
    assert len(inputs) == len(attributions), (
        """The number of tensors in the inputs and
        attributions must match. Found number of tensors in the inputs is: {} and in the
        attributions: {}"""
    ).format(len(inputs), len(attributions))

    bsz = inputs[0].size(0)
    faithfulness_estimate_batch = []
    for sample_idx in range(bsz):
        faithfulness_estimate = (
            eval_faithfulness_estimate_single_sample_tupled_computation(
                forward_func=forward_func,
                inputs=tuple(input[sample_idx].unsqueeze(0) for input in inputs),
                attributions=tuple(
                    attr[sample_idx].unsqueeze(0) for attr in attributions
                ),
                feature_masks=(
                    tuple(mask[sample_idx].unsqueeze(0) for mask in feature_masks)
                    if feature_masks is not None
                    else None
                ),
                baselines=tuple(
                    baseline[sample_idx].unsqueeze(0) for baseline in baselines
                ),
                additional_forward_args=(
                    x[sample_idx].unsqueeze(0) for x in additional_forward_args
                ),
                target=target[sample_idx],
                max_examples_per_batch=max_examples_per_batch,
            )
        )
        faithfulness_estimate_batch.append(faithfulness_estimate)
    faithfulness_estimate_batch = torch.tensor(faithfulness_estimate_batch)
    return faithfulness_estimate_batch


@log_usage()
def eval_faithfulness_estimate_single_sample(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType,
    feature_masks: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_examples_per_batch: int = None,
) -> Tensor:
    def _next_faithfulness_estimate_tensors(
        current_n_perturbed_features: int,
        current_n_steps: int,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]:
        # get the indices of features that will be perturbed in the current iteration
        # for example if we do 1 by 1 perturbation, then the first iteration will perturb the first feature
        # the second iteration will perturb both the first and second feature and so on
        inputs_perturbed = inputs.repeat(
            current_n_perturbed_features, *tuple([1] * len(inputs.shape[1:]))
        )
        attributions_sum_perturbed = []
        for perturbation_sample_idx, feature_idx in enumerate(
            range(current_n_steps - current_n_perturbed_features, current_n_steps)
        ):
            # for each feature in the current step incrementally replace the baseline with the original sample
            perturbation_mask = (
                feature_masks == descending_attribution_indices[feature_idx]
            )
            inputs_perturbed[perturbation_sample_idx][perturbation_mask[0]] = baselines[
                perturbation_mask
            ]  # input[0] here since batch size is 1
            attributions_sum_perturbed.append((attributions * perturbation_mask).sum())

        attributions_sum_perturbed = torch.stack(attributions_sum_perturbed)
        targets_expanded = _expand_target(
            target,
            current_n_perturbed_features,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        additional_forward_args_expanded = _expand_additional_forward_args(
            additional_forward_args,
            current_n_perturbed_features,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        inputs_perturbed_fwd = _run_forward(
            forward_func,
            # the inputs are [batch_size, feature_size, feature_dims] so we need to split them by feature size
            _split_tensors_to_tuple_tensors(inputs_perturbed, inputs_shape, dim=1),
            targets_expanded,
            additional_forward_args_expanded,
        )
        inputs_perturbed_fwd_diff = inputs_fwd - inputs_perturbed_fwd

        # this is a list of tensors which holds the forward outputs of the model
        # on each feature group perturbation
        # the first element will be when the feature with lowest importance is added to the baseline
        # the last element will be when all features are added to the baseline
        return (
            list(inputs_perturbed_fwd_diff.numpy()),
            list(attributions_sum_perturbed.numpy()),
        )

    def _sum_faithfulness_estimate_tensors(agg_tensors, tensors):
        return tuple(agg_t + t for agg_t, t in zip(agg_tensors, tensors))

    bsz = inputs[0].size(0)
    assert bsz == 1, "Batch size must be 1 for faithfulness_estimate_single_sample"

    # get the first input forward output
    inputs_fwd = _run_forward(forward_func, inputs, target, additional_forward_args)

    # flatten all inputs and baseline features in the input
    inputs, inputs_shape = _tuple_tensors_to_tensors(inputs)
    baselines, baselines_shape = _tuple_tensors_to_tensors(baselines)
    assert (
        inputs_shape == baselines_shape
    ), "Inputs and baselines must have the same shape"

    # flatten all feature masks in the input
    if feature_masks is not None:
        feature_masks, _ = _tuple_tensors_to_tensors(feature_masks)
    else:
        feature_masks = _construct_default_feature_masks(attributions)
        feature_masks, _ = _tuple_tensors_to_tensors(feature_masks)

    # flatten all attributions in the input, this must be done after the feature masks are flattened as
    # feature masks may depened on attribution
    attributions, _ = _tuple_tensors_to_tensors(attributions)

    # validate feature masks are increasing non-negative
    _validate_feature_mask(feature_masks)

    # gather attribution scores of feature groups
    # this can be useful for efficiently summing up attributions of feature groups
    # this is why we need a single batch size as gathered attributes and number of features for each
    # sample can be different
    reduced_attributions, n_features = _reduce_tensor_with_indices(
        attributions[0], indices=feature_masks[0].flatten()
    )

    # get the gathererd-attributions sorted in descending order of their importance
    descending_attribution_indices = torch.argsort(
        reduced_attributions, descending=True
    )

    with torch.no_grad():
        # the logic for this implementation as as follows:
        # we start from baseline and in each iteration, a feature group is replaced by the original sample
        # in ascending order of its importance
        agg_tensors = _divide_and_aggregate_metrics_n_features(
            n_features,
            _next_faithfulness_estimate_tensors,
            agg_func=_sum_faithfulness_estimate_tensors,
            max_examples_per_batch=max_examples_per_batch,
        )

        agg_tensors = tuple(np.array(x).flatten() for x in agg_tensors)
        inputs_perturbed_fwd_diffs = agg_tensors[0]
        attributions_sum_perturbed = agg_tensors[1]
        faithfulness_estimate_score = scipy.stats.pearsonr(
            inputs_perturbed_fwd_diffs, attributions_sum_perturbed
        )[0]

    return (
        faithfulness_estimate_score,
        attributions_sum_perturbed,
        inputs_perturbed_fwd_diffs,
    )


@log_usage()
def faithfulness_estimate(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType,
    feature_masks: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_examples_per_batch: int = None,
) -> Tensor:
    # perform argument formattings
    inputs = _format_tensor_into_tuples(inputs)  # type: ignore
    if baselines is None:
        baselines = tuple(torch.zeros_like(inp) for inp in inputs)
    else:
        baselines = _format_baseline(baselines, cast(Tuple[Tensor, ...], inputs))
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    attributions = _format_tensor_into_tuples(attributions)  # type: ignore

    # Make sure that inputs and corresponding attributions have matching sizes.
    assert len(inputs) == len(attributions), (
        """The number of tensors in the inputs and
        attributions must match. Found number of tensors in the inputs is: {} and in the
        attributions: {}"""
    ).format(len(inputs), len(attributions))

    bsz = inputs[0].size(0)
    faithfulness_estimate_batch = []
    attributions_sum_perturbed_batch = []
    inputs_perturbed_fwd_diffs_batch = []
    for sample_idx in range(bsz):
        (
            faithfulness_estimate,
            attributions_sum_perturbed,
            inputs_perturbed_fwd_diffs,
        ) = eval_faithfulness_estimate_single_sample(
            forward_func=forward_func,
            inputs=tuple(input[sample_idx].unsqueeze(0) for input in inputs),
            attributions=tuple(attr[sample_idx].unsqueeze(0) for attr in attributions),
            feature_masks=(
                tuple(mask[sample_idx].unsqueeze(0) for mask in feature_masks)
                if feature_masks is not None
                else None
            ),
            baselines=(
                tuple(baseline[sample_idx].unsqueeze(0) for baseline in baselines)
                if baselines is not None
                else None
            ),
            additional_forward_args=(
                tuple(
                    (
                        arg[sample_idx].unsqueeze(0)
                        if isinstance(arg, torch.Tensor)
                        else arg
                    )
                    for arg in additional_forward_args
                )
                if additional_forward_args is not None
                else None
            ),
            target=target[sample_idx] if target is not None else None,
            max_examples_per_batch=max_examples_per_batch,
        )
        faithfulness_estimate_batch.append(faithfulness_estimate)
        attributions_sum_perturbed_batch.append(attributions_sum_perturbed)
        inputs_perturbed_fwd_diffs_batch.append(inputs_perturbed_fwd_diffs)
    faithfulness_estimate_batch = torch.tensor(faithfulness_estimate_batch)
    attributions_sum_perturbed_batch = torch.tensor(attributions_sum_perturbed_batch)
    inputs_perturbed_fwd_diffs_batch = torch.tensor(inputs_perturbed_fwd_diffs_batch)
    return (
        faithfulness_estimate_batch,
        attributions_sum_perturbed_batch,
        inputs_perturbed_fwd_diffs_batch,
    )
