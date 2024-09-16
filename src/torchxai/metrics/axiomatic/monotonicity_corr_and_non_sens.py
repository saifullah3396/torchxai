#!/usr/bin/env python3

from typing import Any, Callable, Tuple, Union, cast

import numpy as np
import scipy
import torch
from captum._utils.common import (
    ExpansionTypes,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_tensor_into_tuples,
    _run_forward,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.log import log_usage
from torch import Tensor

from torchxai.metrics._utils.batching import (
    _divide_and_aggregate_metrics_n_perturbations_per_feature,
)
from torchxai.metrics._utils.common import (
    _construct_default_feature_masks,
    _validate_feature_mask,
)
from torchxai.metrics._utils.perturbation import default_perturb_func


@log_usage()
def eval_monotonicity_corr_and_non_sens_single_sample(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    feature_masks: TensorOrTupleOfTensorsGeneric = None,
    n_perturbations_per_feature: int = 10,
    additional_forward_args: Any = None,
    target: TargetType = None,
    perturb_func: Callable = default_perturb_func(),
    max_examples_per_batch: int = None,
    eps: float = 1e-5,
) -> Tensor:
    def _generate_perturbations(
        current_n_perturbed_features: int, current_feature_indices: int, feature_masks
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
        r"""
        The perturbations are generated for each example
        `current_n_perturbed_features` times.

        For performance reasons we are not calling `perturb_func` on each example but
        on a batch that contains `current_n_perturbed_features`
        repeated instances per examp_generate_perturbationsle.
        """

        def call_perturb_func():
            r""" """
            inputs_pert: Union[Tensor, Tuple[Tensor, ...]]
            if len(inputs_expanded) == 1:
                inputs_pert = inputs_expanded[0]
                feature_masks_pert = feature_masks_expanded[0]
            else:
                inputs_pert = inputs_expanded
                feature_masks_pert = feature_masks_expanded
            return perturb_func(
                inputs=inputs_pert, perturbation_masks=feature_masks_pert
            )

        # repeat each current_n_perturbed_features times
        inputs_expanded = tuple(
            input.repeat(
                n_perturbations_per_feature * current_n_perturbed_features,
                *tuple([1] * len(input.shape[1:]))
            )
            for input in inputs
        )

        feature_masks_expanded = tuple(
            torch.cat(
                [(mask == feature_index) for feature_index in current_feature_indices]
            ).repeat_interleave(repeats=n_perturbations_per_feature, dim=0)
            for mask in feature_masks
        )
        return call_perturb_func()

    def _validate_inputs_and_perturbations(
        inputs: Tuple[Tensor, ...],
        inputs_perturbed: Tuple[Tensor, ...],
    ) -> None:
        # asserts the sizes of the perturbations and inputs
        assert len(inputs_perturbed) == len(inputs), (
            """The number of perturbed
            inputs and corresponding perturbations must have the same number of
            elements. Found number of inputs is: {} and perturbations:
            {}"""
        ).format(len(inputs_perturbed), len(inputs))

        # asserts the shapes of the perturbations and perturbed inputs
        for inputs, input_perturbed in zip(inputs, inputs_perturbed):
            assert inputs[0].shape == input_perturbed[0].shape, (
                """Perturbed input
                and corresponding perturbation must have the same shape and
                dimensionality. Found perturbation shape is: {} and the input shape
                is: {}"""
            ).format(inputs[0].shape, input_perturbed[0].shape)

    def _next_monotonicity_corr_tensors(
        current_n_perturbed_features: int,
        current_n_steps: int,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]:
        current_feature_indices = torch.arange(
            current_n_steps - current_n_perturbed_features, current_n_steps
        )
        inputs_perturbed = _generate_perturbations(
            current_n_perturbed_features, current_feature_indices, feature_masks
        )
        inputs_perturbed = _format_tensor_into_tuples(inputs_perturbed)
        _validate_inputs_and_perturbations(
            cast(Tuple[Tensor, ...], inputs),
            cast(Tuple[Tensor, ...], inputs_perturbed),
        )
        additional_forward_args_expanded = _expand_additional_forward_args(
            additional_forward_args,
            n_perturbations_per_feature * current_n_perturbed_features,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        targets_expanded = _expand_target(
            target,
            current_n_perturbed_features,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        inputs_fwd = _run_forward(forward_func, inputs, target, additional_forward_args)
        inputs_fwd_inv = 1.0 if np.abs(inputs_fwd) < eps else 1.0 / np.abs(inputs_fwd)
        inputs_perturbed_fwd = _run_forward(
            forward_func,
            inputs_perturbed,
            targets_expanded,
            additional_forward_args_expanded,
        )
        perturbed_fwd_diffs = inputs_perturbed_fwd - inputs_fwd
        perturbed_fwd_diffs = perturbed_fwd_diffs.chunk(current_n_perturbed_features)

        # each element in the tuple corresponds to a single feature group
        curr_perturbed_fwd_diffs_relative_vars = tuple(
            torch.mean(x**2) * (inputs_fwd_inv**2) for x in perturbed_fwd_diffs
        )

        # gather the attribution scores of all the features in the current feature group
        curr_feature_attribution_scores = tuple(
            torch.cat(
                [
                    attribution[mask == feature_index]
                    for attribution, mask in zip(attributions, feature_masks)
                ]
            )
            for feature_index in current_feature_indices
        )

        # compute the monotonicity corr for each feature group
        curr_feature_group_attribution_scores = tuple(
            x.sum() for x in curr_feature_attribution_scores
        )

        # return the perturbed forward differences and the current feature attribution scores
        curr_perturbed_fwd_diffs_relative_vars = tuple(
            x.item() for x in curr_perturbed_fwd_diffs_relative_vars
        )
        curr_feature_group_attribution_scores = tuple(
            x.item() for x in curr_feature_group_attribution_scores
        )
        return list(curr_perturbed_fwd_diffs_relative_vars), list(
            curr_feature_group_attribution_scores
        )

    def _sum_monotonicity_corr_tensors(agg_tensors, tensors):
        return tuple(agg_t + t for agg_t, t in zip(agg_tensors, tensors))

    with torch.no_grad():
        bsz = inputs[0].size(0)
        assert bsz == 1, "Batch size must be 1 for monotonicity_corr_single_sample"
        if feature_masks is not None:
            # assert that all elements in the feature_masks are unique and non-negative increasing
            _validate_feature_mask(feature_masks)
        else:
            feature_masks = _construct_default_feature_masks(attributions)

        # this assumes a batch size of 1, this will not work for batch size > 1
        n_features = max(x.max() for x in feature_masks).item() + 1

        # the logic for this implementation as as follows:
        # each input is repeated n_perturbations_per_feature times to create an input of shape
        # tuple(
        #   [(n_perturbations_per_feature, input.shape[0], ...), (n_perturbations_per_feature, input.shape[0], ...)]
        # )
        # then in each perturbation step, every feature is sequentually perturbed n_perturbations_per_feature times
        # so perturbation step will have the n_pertrubations_per_feature perturbations for the first feature, then
        # n_perturbations_per_feature perturbations for the second feature and so on.
        # and the total perturbation steps n_features are equal to total features that need to be perturbed
        # in case of feature_masks, n_features is the total number of feature groups present in the inputs
        # each group is perturbed together n_perturbations_per_feature times
        # in case there is no feature masks, then a corresponding feature mask is generated for each input feature
        agg_tensors = _divide_and_aggregate_metrics_n_perturbations_per_feature(
            n_perturbations_per_feature,
            n_features,
            _next_monotonicity_corr_tensors,
            agg_func=_sum_monotonicity_corr_tensors,
            max_examples_per_batch=max_examples_per_batch,
        )

        # compute monotonocity corr metric
        def compute_monotonocity_corr(
            perturbed_fwd_diffs_relative_vars: np.ndarray,
            feature_group_attribution_scores: np.ndarray,
        ):
            # get the attribution scores ascending order indices
            ascending_attribution_indices = np.argsort(feature_group_attribution_scores)

            # sort the attribution scores and the model forward variances by the ascending order of the attribution scores
            ascending_attribution_scores = feature_group_attribution_scores[
                ascending_attribution_indices
            ]
            ascending_attribution_perturbed_fwd_diffs_relative_vars = (
                perturbed_fwd_diffs_relative_vars[ascending_attribution_indices]
            )

            # find the spearman corr between the ascending order of the attribution scores and the model forward variances
            # this corr should be close to 1 if the model forward variances are monotonically increasing with the attribution scores
            # this means that features that have a lower attribution score are directly correlated with lower effect on the model output
            return scipy.stats.spearmanr(
                ascending_attribution_scores,
                ascending_attribution_perturbed_fwd_diffs_relative_vars,
            )[0]

        # compute non-sensitivity metric
        def compute_non_sens(
            perturbed_fwd_diffs_relative_vars: np.ndarray,
            feature_group_attribution_scores: np.ndarray,
        ):
            # find the indices of features that have a zero attribution score, every attribution score value less
            # than non_sens_eps is considered zero
            zero_attribution_features = set(
                list(
                    np.argwhere(
                        np.abs(feature_group_attribution_scores) < eps
                    ).flatten()
                )
            )
            zero_variance_features = set(
                list(
                    np.argwhere(
                        np.abs(perturbed_fwd_diffs_relative_vars) < eps
                    ).flatten()
                )
            )

            # find the symmetric difference of the zero attribution features and the zero variance features
            # this set should be empty if the model is non-sensitive to the zero attribution features
            # symmetric difference will give the oppposite of the intersection of the two sets
            # therefore non-sensitivity corresponds to the number of features that have either:
            # 1. zero attribution scores and non-zero model forward variances
            # 2. non-zero attribution scores and zero model forward variances
            # a higher non-sensitivity score indicates that the model is more sensitive to the zero attribution features
            # and a lower non-sensitivity score indicates that the model is non-sensitive to the zero attribution features
            return len(
                zero_attribution_features.symmetric_difference(zero_variance_features)
            )

        agg_tensors = tuple(np.array(x) for x in agg_tensors)
        perturbed_fwd_diffs_relative_vars = agg_tensors[0]
        feature_group_attribution_scores = agg_tensors[1]

        monotonicity_corr = compute_monotonocity_corr(
            perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores
        )
        non_sens = compute_non_sens(
            perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores
        )
    return monotonicity_corr, non_sens


@log_usage()
def monotonicity_corr_and_non_sens(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    feature_masks: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    perturb_func: Callable = default_perturb_func(),
    n_perturbations_per_feature: int = 10,
    max_examples_per_batch: int = None,
) -> Tensor:
    with torch.no_grad():
        # perform argument formattings
        inputs = _format_tensor_into_tuples(inputs)  # type: ignore
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        attributions = _format_tensor_into_tuples(attributions)  # type: ignore

        # Make sure that inputs and corresponding attributions have matching sizes.
        assert len(inputs) == len(attributions), (
            """The number of tensors in the inputs and
            attributions must match. Found number of tensors in the inputs is: {} and in the
            attributions: {}"""
        ).format(len(inputs), len(attributions))

        bsz = inputs[0].size(0)
        monotonicity_corr_batch = []
        non_sens_batch = []
        for sample_idx in range(bsz):
            monotonicity_corr, non_sens = (
                eval_monotonicity_corr_and_non_sens_single_sample(
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
                    perturb_func=perturb_func,
                    n_perturbations_per_feature=n_perturbations_per_feature,
                    max_examples_per_batch=max_examples_per_batch,
                )
            )

            monotonicity_corr_batch.append(monotonicity_corr)
            non_sens_batch.append(non_sens)
        monotonicity_corr_batch = torch.tensor(monotonicity_corr_batch)
        non_sens_batch = torch.tensor(non_sens_batch)
        return monotonicity_corr_batch, non_sens_batch
