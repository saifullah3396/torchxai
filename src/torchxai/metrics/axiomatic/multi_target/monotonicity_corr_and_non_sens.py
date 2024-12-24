import inspect
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
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor
from torchxai.explainers._utils import _run_forward_multi_target
from torchxai.metrics._utils.batching import (
    _divide_and_aggregate_metrics_n_perturbations_per_feature,
)
from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _feature_mask_to_chunked_perturbation_mask_with_attributions_list,
    _reduce_tensor_with_indices_non_deterministic,
    _split_tensors_to_tuple_tensors,
    _tuple_tensors_to_tensors,
    _validate_feature_mask,
)
from torchxai.metrics._utils.perturbation import default_fixed_baseline_perturb_func


def _eval_mutli_target_monotonicity_corr_and_non_sens_single_sample(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions_list: List[TensorOrTupleOfTensorsGeneric],
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    targets_list: List[TargetType] = None,
    frozen_features: Optional[List[torch.Tensor]] = None,
    perturb_func: Callable = default_fixed_baseline_perturb_func(),
    n_perturbations_per_feature: int = 10,
    max_features_processed_per_batch: int = None,
    percentage_feature_removal_per_step: float = 0.0,
    zero_attribution_threshold: float = 0.01,
    zero_variance_threshold: float = 0.01,
    use_percentage_attribution_threshold: bool = True,
    return_ratio: bool = True,
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
            baselines_pert = None
            inputs_pert: Union[Tensor, Tuple[Tensor, ...]]
            if len(inputs_expanded) == 1:
                inputs_pert = inputs_expanded[0]
                perturbation_masks = perturbation_mask_expanded[0]
                if baselines_expanded is not None:
                    baselines_pert = cast(Tuple, baselines_expanded)[0]
            else:
                inputs_pert = inputs_expanded
                perturbation_masks = perturbation_mask_expanded
                baselines_pert = baselines_expanded

            valid_args = inspect.signature(perturb_func).parameters.keys()
            perturb_kwargs = dict(
                inputs=inputs_pert,
                perturbation_masks=perturbation_masks,
            )
            if "baselines" in valid_args:
                assert (
                    baselines_pert is not None
                ), f"""The perturb_func {perturb_func} requires baselines as an argument. Please provide baselines."""
                perturb_kwargs["baselines"] = baselines_pert
            return perturb_func(**perturb_kwargs)

        # repeat each current_n_perturbed_features times
        inputs_expanded = tuple(
            input.repeat(
                n_perturbations_per_feature * current_n_perturbed_features,
                *tuple([1] * len(input.shape[1:])),
            )
            for input in inputs
        )

        baselines_expanded = baselines
        if baselines is not None:
            baselines_expanded = tuple(
                (
                    baseline.repeat_interleave(
                        n_perturbations_per_feature * current_n_perturbed_features,
                        dim=0,
                    )
                    if isinstance(baseline, torch.Tensor)
                    and baseline.shape[0] == input.shape[0]
                    else baseline
                )
                for input, baseline in zip(inputs, cast(Tuple, baselines))
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
            current_n_steps - current_n_perturbed_features,
            current_n_steps,
            device=inputs[0].device,
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
                if torch.abs(inputs_fwd[:, i]) < 1e-8
                else 1.0 / torch.abs(inputs_fwd[:, i])
            )
            for i in range(inputs_fwd.shape[1])
        ]
        return [
            [
                (torch.mean(x**2) * (inputs_fwd_inv**2)).item()
                for x in perturbed_fwd_diffs
            ]
            for perturbed_fwd_diffs, inputs_fwd_inv in zip(
                perturbed_fwd_diffs_list, inputs_fwd_inv_list
            )
        ]

    def _agg_monotonicity_corr_tensors(agg_tensors_list, tensors_list):
        return [
            agg_tensors + tensors
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

        # remove the batch index
        feature_mask_flattened = feature_mask_flattened.squeeze()

        def reduce_attributions_in_list(attributions):
            # flatten all attributions in the input, this must be done after the feature masks are flattened as
            # feature masks may depened on attribution
            attributions, _ = _tuple_tensors_to_tensors(attributions)

            # in this step we reduce the attributions to the feature groups
            # here the weighted sum of the attributions is computed for each feature group
            # the 0th index of reduced_attributions corresponds to the 0th feature group
            # the 1th index of reduced_attributions corresponds to the 1th feature group and so on
            # therefore be careful that the order of the original feature group is not preserved
            # for example, the feature mask may have the indices [0, 3, 4, 5, 1] but the reduced attributions
            # may have the indices [0, 1, 2, 3, 4] where the 0th index of the reduced attributions corresponds to the
            # 0th index of the feature mask
            reduced_attributions, _ = _reduce_tensor_with_indices_non_deterministic(
                attributions[0], indices=feature_mask_flattened
            )

            # after reduction we take the absolute value of the attributions as we are interested in the
            # total magnitude of the attributions
            return reduced_attributions.abs()

        reduced_attributions_list = [
            reduce_attributions_in_list(attributions)
            for attributions in attributions_list
        ]

        # generate the feature_indices
        feature_indices = torch.arange(
            reduced_attributions_list[0].size(0), device=inputs[0].device
        )

        # now we generate the global perturbation masks and the chunked reduced attributions
        # each step of the perturbation will generate a perturbation mask for all the features in a single chunk
        # the chunk size is determined by the percentage_feature_removal_per_step
        # if the percentage_feature_removal_per_step is 0.01, then 1% of the features will be removed in each step
        # if the percentage_feature_removal_per_step is 0, then chunk size is set to 1 in which case every
        # perturbation step will remove a single feature in the ascending order of importance
        # the returned chunk_reduced_attributions will have the same shape as the global_perturbation_masks
        # and will contain the sum of attributions over the features in the each chunk
        global_perturbation_masks, chunk_reduced_attributions_list = (
            _feature_mask_to_chunked_perturbation_mask_with_attributions_list(
                feature_mask_flattened,
                reduced_attributions_list,
                feature_indices,
                frozen_features,
                percentage_feature_removal_per_step,
            )
        )
        assert (
            global_perturbation_masks.shape[0]
            == chunk_reduced_attributions_list[0].shape[0]
        )

        # draw all the global_perturbation_masks for debugging
        # global_perturbation_masks_reshaped = _split_tensors_to_tuple_tensors(
        #     global_perturbation_masks, flattened_mask_shape
        # )
        # _draw_perturbated_inputs_sequences_images(global_perturbation_masks_reshaped)

        # note that the if frozen_features are present, they are not perturbed, and therefore the
        # total global_perturbation_masks size may be less than the total number of features
        # therefore we reassign the n_features to the total number of features that are perturbed
        n_features = global_perturbation_masks.shape[0]

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
        agg_tensors = _divide_and_aggregate_metrics_n_perturbations_per_feature(
            n_perturbations_per_feature,
            global_perturbation_masks.shape[0],
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
                feature_group_attribution_scores,
                perturbed_fwd_diffs_relative_vars,
            )[0]

        # compute non-sensitivity metric
        def compute_non_sens(
            perturbed_fwd_diffs_relative_vars: np.ndarray,
            feature_group_attribution_scores: np.ndarray,
        ):
            # find the indices of features that have a zero attribution score, every attribution score value less
            # than non_sens_eps is considered zero
            def find_small_scale_features(
                values: np.ndarray, threshold: Optional[float] = None
            ):
                return set(list(np.argwhere(np.abs(values) < threshold).flatten()))

            if use_percentage_attribution_threshold:
                feature_group_attribution_scores = (
                    feature_group_attribution_scores
                    / np.sum(feature_group_attribution_scores)
                )

            # find the indices of features that have a zero attribution score.
            # All values below the threshold are considered zero, default threshold is set to 1% of the max value
            # This can be sensitive to the max value of the attribution scores but since different explanation methods
            # have different scales, it is better to use a relative threshold
            zero_attribution_features = find_small_scale_features(
                feature_group_attribution_scores,
                threshold=zero_attribution_threshold,
            )

            # find the indices of features that have a zero model forward variance,
            # all values below the threshold are considered zero. Default threshold is set to 1%
            zero_variance_features = find_small_scale_features(
                perturbed_fwd_diffs_relative_vars,
                threshold=zero_variance_threshold,
            )

            # find the symmetric difference of the zero attribution features and the zero variance features
            # this set should be empty if the model is non-sensitive to the zero attribution features
            # symmetric difference will give the oppposite of the intersection of the two sets
            # therefore non-sensitivity corresponds to the number of features that have either:
            # 1. zero attribution scores and non-zero model forward variances
            # 2. non-zero attribution scores and zero model forward variances
            # a higher non-sensitivity score indicates that the model is more sensitive to the zero attribution features
            # and a lower non-sensitivity score indicates that the model is non-sensitive to the zero attribution features
            non_sens = len(
                zero_attribution_features.symmetric_difference(zero_variance_features)
            )
            if return_ratio:
                return non_sens / n_features
            return non_sens

        perturbed_fwd_diffs_relative_vars_list = np.array(agg_tensors)
        chunk_reduced_attributions_list = [
            x.cpu().numpy() for x in chunk_reduced_attributions_list
        ]
        monotonicity_corr_list = [
            compute_monotonocity_corr(
                perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores
            )
            for perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores in zip(
                perturbed_fwd_diffs_relative_vars_list,
                chunk_reduced_attributions_list,
            )
        ]
        non_sens_list = [
            compute_non_sens(
                perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores
            )
            for perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores in zip(
                perturbed_fwd_diffs_relative_vars_list,
                chunk_reduced_attributions_list,
            )
        ]
    return (
        monotonicity_corr_list,
        non_sens_list,
        [n_features] * len(monotonicity_corr_list),
        perturbed_fwd_diffs_relative_vars_list,
        chunk_reduced_attributions_list,
    )


def _multi_target_monotonicity_corr_and_non_sens(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions_list: List[TensorOrTupleOfTensorsGeneric],
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    targets_list: List[TargetType] = None,
    frozen_features: Optional[List[torch.Tensor]] = None,
    perturb_func: Callable = default_fixed_baseline_perturb_func(),
    n_perturbations_per_feature: int = 10,
    max_features_processed_per_batch: int = None,
    percentage_feature_removal_per_step: float = 0.0,
    zero_attribution_threshold: float = 1e-5,
    zero_variance_threshold: float = 1e-5,
    use_percentage_attribution_threshold: bool = False,
    return_ratio: bool = True,
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
        n_features_list_batch = []
        perturbed_fwd_diffs_relative_vars_list_batch = []
        feature_group_attribution_scores_list_batch = []
        for sample_idx in tqdm.tqdm(range(bsz), disable=not show_progress):
            (
                monotonicity_corr_list,
                non_sens_list,
                n_features_list,
                perturbed_fwd_diffs_relative_vars_list,
                feature_group_attribution_scores_list,
            ) = _eval_mutli_target_monotonicity_corr_and_non_sens_single_sample(
                forward_func=forward_func,
                inputs=tuple(input[sample_idx].unsqueeze(0) for input in inputs),
                attributions_list=[
                    tuple(attr[sample_idx].unsqueeze(0) for attr in attributions)
                    for attributions in attributions_list
                ],
                baselines=(
                    tuple(baseline[sample_idx].unsqueeze(0) for baseline in baselines)
                    if baselines is not None
                    else None
                ),
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
                frozen_features=(
                    frozen_features[sample_idx]
                    if frozen_features is not None
                    else frozen_features
                ),
                perturb_func=perturb_func,
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
                percentage_feature_removal_per_step=percentage_feature_removal_per_step,
                zero_attribution_threshold=zero_attribution_threshold,
                zero_variance_threshold=zero_variance_threshold,
                use_percentage_attribution_threshold=use_percentage_attribution_threshold,
                return_ratio=return_ratio,
                show_progress=show_progress,
            )

            monotonicity_corr_list_batch.append(monotonicity_corr_list)
            non_sens_list_batch.append(non_sens_list)
            n_features_list_batch.append(n_features_list)
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
        n_features_list_batch = [
            torch.tensor(x) for x in list(zip(*n_features_list_batch))
        ]
        return (
            monotonicity_corr_batch_list,
            non_sens_batch_list,
            n_features_list_batch,
            perturbed_fwd_diffs_relative_vars_batch_list,
            feature_group_attribution_scores_batch_list,
        )
