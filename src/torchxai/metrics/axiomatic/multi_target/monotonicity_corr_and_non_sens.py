#!/usr/bin/env python3

from typing import Any, Callable, List, Optional, Tuple, Union, cast

import numpy as np
import scipy
import torch
import tqdm
from captum._utils.common import (
    ExpansionTypes,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_tensor_into_tuples,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor
from torchxai.explainers._utils import _run_forward_multi_target
from torchxai.metrics._utils.batching import (
    _divide_and_aggregate_metrics_n_perturbations_per_feature,
)
from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _split_tensors_to_tuple_tensors,
    _tuple_tensors_to_tensors,
    _validate_feature_mask,
)
from torchxai.metrics._utils.perturbation import default_fixed_baseline_perturb_func


def _eval_mutli_target_monotonicity_corr_and_non_sens_single_sample(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions_list: List[TensorOrTupleOfTensorsGeneric],
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    targets_list: List[TargetType] = None,
    perturb_func: Callable = default_fixed_baseline_perturb_func(),
    n_perturbations_per_feature: int = 10,
    max_features_processed_per_batch: int = None,
    frozen_features: Optional[List[torch.Tensor]] = None,
    eps: float = 0.01,
    show_progress: bool = False,
) -> Tensor:
    def _generate_perturbations(
        current_n_perturbed_features: int, current_perturbation_mask: Tensor
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
        r"""
        The perturbations are generated for each example
        `current_n_perturbed_features * n_perturbations_per_feature` times.

        For performance reasons we are not calling `perturb_func` on each example but
        on a batch that contains `current_n_perturbed_features * n_perturbations_per_feature`
        repeated instances per example.
        """

        def call_perturb_func():
            r""" """
            inputs_pert: Union[Tensor, Tuple[Tensor, ...]]
            if len(inputs_expanded) == 1:
                inputs_pert = inputs_expanded[0]
                perturbation_masks = perturbation_mask_expanded[0]
            else:
                inputs_pert = inputs_expanded
                perturbation_masks = perturbation_mask_expanded
            return perturb_func(
                inputs=inputs_pert, perturbation_masks=perturbation_masks
            )

        # repeat each current_n_perturbed_features times
        inputs_expanded = tuple(
            input.repeat(
                n_perturbations_per_feature * current_n_perturbed_features,
                *tuple([1] * len(input.shape[1:])),
            )
            for input in inputs
        )

        # repeat each perturbation mask n_perturbations_per_feature times
        perturbation_mask_expanded = current_perturbation_mask.repeat_interleave(
            repeats=n_perturbations_per_feature, dim=0
        )

        # split back to tuple tensors
        perturbation_mask_expanded = _split_tensors_to_tuple_tensors(
            perturbation_mask_expanded, flattened_mask_shape
        )

        # view as input shape (this is only necessary for edge cases where input is of (1, 1) shape)
        perturbation_mask_expanded = tuple(
            mask.view_as(input)
            for mask, input in zip(perturbation_mask_expanded, inputs_expanded)
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
            current_n_perturbed_features,
            global_perturbation_masks[current_feature_indices],
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
        targets_expanded_list = [
            _expand_target(
                target,
                current_n_perturbed_features,
                expansion_type=ExpansionTypes.repeat_interleave,
            )
            for target in targets_list
        ]
        inputs_fwd = _run_forward_multi_target(
            forward_func, inputs, targets_expanded_list, additional_forward_args
        )
        inputs_perturbed_fwd = _run_forward_multi_target(
            forward_func,
            inputs_perturbed,
            targets_expanded_list,
            additional_forward_args_expanded,
        )
        perturbed_fwd_diffs = inputs_fwd - inputs_perturbed_fwd

        # make perturbed_fwd_diffs_list list
        perturbed_fwd_diffs_list = [
            perturbed_fwd_diffs[:, i] for i in range(perturbed_fwd_diffs.shape[1])
        ]
        perturbed_fwd_diffs_list = [
            perturbed_fwd_diffs.chunk(current_n_perturbed_features)
            for perturbed_fwd_diffs in perturbed_fwd_diffs_list
        ]

        inputs_fwd_inv_list = [
            (
                1.0
                if torch.abs(inputs_fwd[:, i]) < eps
                else 1.0 / torch.abs(inputs_fwd[:, i])
            )
            for i in range(inputs_fwd.shape[1])
        ]
        # each element in the tuple corresponds to a single feature group
        curr_perturbed_fwd_diffs_relative_vars_list = [
            tuple(torch.mean(x**2) * (inputs_fwd_inv**2) for x in perturbed_fwd_diffs)
            for perturbed_fwd_diffs, inputs_fwd_inv in zip(
                perturbed_fwd_diffs_list, inputs_fwd_inv_list
            )
        ]

        # gather the attribution scores of all the features in the current feature group
        curr_feature_attribution_scores_list = [
            tuple(
                torch.cat(
                    [
                        attribution[mask == feature_index]
                        for attribution, mask in zip(attributions, feature_mask)
                    ]
                )
                for feature_index in current_feature_indices
            )
            for attributions in attributions_list
        ]

        # compute the monotonicity corr for each feature group
        # here we sum the attributions over feature groups, but should we sum the absolutes?
        # this question is not clear from the paper
        curr_feature_group_attribution_scores_list = [
            tuple(x.sum() for x in curr_feature_attribution_scores)
            for curr_feature_attribution_scores in curr_feature_attribution_scores_list
        ]

        # return the perturbed forward differences and the current feature attribution scores
        curr_perturbed_fwd_diffs_relative_vars_list = [
            tuple(x.item() for x in curr_perturbed_fwd_diffs_relative_vars)
            for curr_perturbed_fwd_diffs_relative_vars in curr_perturbed_fwd_diffs_relative_vars_list
        ]
        curr_feature_group_attribution_scores_list = [
            tuple(x.item() for x in curr_feature_group_attribution_scores)
            for curr_feature_group_attribution_scores in curr_feature_group_attribution_scores_list
        ]
        return (
            curr_perturbed_fwd_diffs_relative_vars_list,
            curr_feature_group_attribution_scores_list,
        )

    def _agg_monotonicity_corr_tensors(agg_tensors_list, tensors_list):
        return [
            tuple(agg_t + t for agg_t, t in zip(agg_tensors, tensors))
            for agg_tensors, tensors in zip(agg_tensors_list, tensors_list)
        ]

    with torch.no_grad():
        bsz = inputs[0].size(0)
        assert bsz == 1, "Batch size must be 1 for monotonicity_corr_single_sample"
        if feature_mask is None:
            feature_mask = _construct_default_feature_mask(attributions_list[0])

        # flatten the feature mask
        feature_mask_flattened, flattened_mask_shape = _tuple_tensors_to_tensors(
            feature_mask
        )

        # validate feature masks are increasing non-negative
        _validate_feature_mask(feature_mask_flattened)

        # this assumes a batch size of 1, this will not work for batch size > 1
        feature_mask_flattened = feature_mask_flattened.squeeze()
        feature_indices = feature_mask_flattened.unique()

        # filter out frozen features if necessary
        if frozen_features is not None:
            mask = ~torch.isin(
                feature_indices,
                torch.tensor(frozen_features, device=inputs[0].device),
            )
            feature_indices = feature_indices[mask]

        n_features = feature_indices.shape[0]
        global_perturbation_masks = feature_mask_flattened.unsqueeze(
            0
        ) == feature_indices.unsqueeze(1)

        # the logic for this implementation as as follows:
        # each input is repeated n_perturbations_per_feature times to create an input of shape
        # tuple(
        #   [(n_perturbations_per_feature, input.shape[0], ...), (n_perturbations_per_feature, input.shape[0], ...)]
        # )
        # then in each perturbation step, every feature is sequentually perturbed n_perturbations_per_feature times
        # so perturbation step will have the n_pertrubations_per_feature perturbations for the first feature, then
        # n_perturbations_per_feature perturbations for the second feature and so on.
        # and the total perturbation steps n_features are equal to total features that need to be perturbed
        # in case of feature_mask, n_features is the total number of feature groups present in the inputs
        # each group is perturbed together n_perturbations_per_feature times
        # in case there is no feature masks, then a corresponding feature mask is generated for each input feature
        agg_tensors_list = _divide_and_aggregate_metrics_n_perturbations_per_feature(
            n_perturbations_per_feature,
            n_features,
            _next_monotonicity_corr_tensors,
            agg_func=_agg_monotonicity_corr_tensors,
            max_features_processed_per_batch=max_features_processed_per_batch,
            show_progress=show_progress,
        )

        # compute monotonocity corr metric
        def compute_monotonocity_corr(
            perturbed_fwd_diffs_relative_vars: np.ndarray,
            feature_group_attribution_scores: np.ndarray,
        ):
            # find the spearman corr between the attribution scores and the model forward variances
            # this corr should be close to 1 if the model forward variances are monotonically increasing with the attribution scores
            # this means that features that have a lower attribution score are directly correlated with lower effect on the model output
            return scipy.stats.spearmanr(
                np.abs(feature_group_attribution_scores),
                perturbed_fwd_diffs_relative_vars,
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

        perturbed_fwd_diffs_relative_vars_list = np.array(agg_tensors_list[0])
        feature_group_attribution_scores_list = np.array(agg_tensors_list[1])

        monotonicity_corr_list = [
            compute_monotonocity_corr(
                perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores
            )
            for perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores in zip(
                perturbed_fwd_diffs_relative_vars_list,
                feature_group_attribution_scores_list,
            )
        ]
        non_sens_list = [
            compute_non_sens(
                perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores
            )
            for perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores in zip(
                perturbed_fwd_diffs_relative_vars_list,
                feature_group_attribution_scores_list,
            )
        ]
    return (
        monotonicity_corr_list,
        non_sens_list,
        n_features,
        perturbed_fwd_diffs_relative_vars_list,
        feature_group_attribution_scores_list,
    )


def _multi_target_monotonicity_corr_and_non_sens(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions_list: List[TensorOrTupleOfTensorsGeneric],
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    targets_list: List[TargetType] = None,
    perturb_func: Callable = default_fixed_baseline_perturb_func(),
    n_perturbations_per_feature: int = 10,
    max_features_processed_per_batch: int = None,
    frozen_features: Optional[List[torch.Tensor]] = None,
    eps: float = 0.01,
    show_progress: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    with torch.no_grad():
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
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        attributions_list = [_format_tensor_into_tuples(attributions) for attributions in attributions_list]  # type: ignore
        feature_mask = _format_tensor_into_tuples(feature_mask)  # type: ignore

        # Make sure that inputs and corresponding attributions have matching sizes.
        assert len(inputs) == len(attributions_list[0]), (
            """The number of tensors in the inputs and
            attributions must match. Found number of tensors in the inputs is: {} and in the
            attributions: {}"""
        ).format(len(inputs), len(attributions_list[0]))
        if feature_mask is not None:
            for mask, attribution in zip(feature_mask, attributions_list[0]):
                assert mask.shape == attribution.shape, (
                    """The shape of the feature mask and the attribution
                    must match. Found feature mask shape: {} and attribution shape: {}"""
                ).format(mask.shape, attribution.shape)

        bsz = inputs[0].size(0)
        monotonicity_corr_list_batch = []
        non_sens_list_batch = []
        n_features_batch = []
        perturbed_fwd_diffs_relative_vars_list_batch = []
        feature_group_attribution_scores_list_batch = []
        for sample_idx in tqdm.tqdm(range(bsz), disable=not show_progress):
            (
                monotonicity_corr_list,
                non_sens_list,
                n_features,
                perturbed_fwd_diffs_relative_vars_list,
                feature_group_attribution_scores_list,
            ) = _eval_mutli_target_monotonicity_corr_and_non_sens_single_sample(
                forward_func=forward_func,
                inputs=tuple(input[sample_idx].unsqueeze(0) for input in inputs),
                attributions_list=[
                    tuple(attr[sample_idx].unsqueeze(0) for attr in attributions)
                    for attributions in attributions_list
                ],
                feature_mask=(
                    tuple(mask[sample_idx].unsqueeze(0) for mask in feature_mask)
                    if feature_mask is not None
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
                targets_list=targets_list,
                perturb_func=perturb_func,
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
                frozen_features=(
                    frozen_features[sample_idx]
                    if frozen_features is not None
                    else frozen_features
                ),
                eps=eps,
                show_progress=show_progress,
            )

            monotonicity_corr_list_batch.append(monotonicity_corr_list)
            non_sens_list_batch.append(non_sens_list)
            n_features_batch.append(n_features)
            perturbed_fwd_diffs_relative_vars_list_batch.append(
                perturbed_fwd_diffs_relative_vars_list
            )
            feature_group_attribution_scores_list_batch.append(
                feature_group_attribution_scores_list
            )

        # convert batch of lists to lists of batches
        monotonicity_corr_batch_list = [
            torch.tensor(x) for x in list(zip(*monotonicity_corr_list_batch))
        ]
        non_sens_batch_list = [torch.tensor(x) for x in list(zip(*non_sens_list_batch))]
        perturbed_fwd_diffs_relative_vars_batch_list = [
            torch.tensor(x)
            for x in list(zip(*perturbed_fwd_diffs_relative_vars_list_batch))
        ]
        feature_group_attribution_scores_batch_list = [
            torch.tensor(x)
            for x in list(zip(*feature_group_attribution_scores_list_batch))
        ]
        n_features_batch = torch.tensor(n_features_batch)
        return (
            monotonicity_corr_batch_list,
            non_sens_batch_list,
            n_features_batch,
            perturbed_fwd_diffs_relative_vars_batch_list,
            feature_group_attribution_scores_batch_list,
        )
