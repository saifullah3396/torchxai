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
    _run_forward,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor
from torchxai.metrics._utils.batching import (
    _divide_and_aggregate_metrics_n_perturbations_per_feature,
)
from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _feature_mask_to_chunked_perturbation_mask_with_attributions,
    _reduce_tensor_with_indices_non_deterministic,
    _split_tensors_to_tuple_tensors,
    _tuple_tensors_to_tensors,
    _validate_feature_mask,
)
from torchxai.metrics._utils.perturbation import default_fixed_baseline_perturb_func
from torchxai.metrics.axiomatic.multi_target.monotonicity_corr_and_non_sens import (
    _multi_target_monotonicity_corr_and_non_sens,
)


def eval_monotonicity_corr_and_non_sens_single_sample(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    frozen_features: Optional[List[torch.Tensor]] = None,
    perturb_func: Callable = default_fixed_baseline_perturb_func(),
    n_perturbations_per_feature: int = 10,
    max_features_processed_per_batch: int = None,
    percentage_feature_removal_per_step: float = 0.0,
    zero_attribution_threshold: float = 0.01,
    zero_variance_threshold: float = 0.01,
    use_percentage_attribution_threshold: bool = True,
    remove_features_in_ascending_order: bool = False,
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
        targets_expanded = _expand_target(
            target,
            current_n_perturbed_features,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        inputs_fwd = _run_forward(forward_func, inputs, target, additional_forward_args)
        inputs_fwd_inv = (
            1.0 if torch.abs(inputs_fwd) < 1e-8 else 1.0 / torch.abs(inputs_fwd)
        )
        inputs_perturbed_fwd = _run_forward(
            forward_func,
            inputs_perturbed,
            targets_expanded,
            additional_forward_args_expanded,
        )
        perturbed_fwd_diffs = inputs_fwd - inputs_perturbed_fwd

        # each element in the tuple corresponds to the multiple perturbations of a single feature group
        perturbed_fwd_diffs = perturbed_fwd_diffs.chunk(current_n_perturbed_features)

        # the relative variance of the perturbed forward differences is computed
        # (std^2 / target_mean^2) where target_mean is the unperturbed input_fwd
        # here std = (sum(perturbed_fwd_diffs^2) / N) where N is the number of perturbations
        return [
            (torch.mean(x**2) * (inputs_fwd_inv**2)).item() for x in perturbed_fwd_diffs
        ]

    def _agg_monotonicity_corr_tensors(agg_tensors, tensors):
        return agg_tensors + tensors

    with torch.no_grad():
        bsz = inputs[0].size(0)
        assert bsz == 1, "Batch size must be 1 for monotonicity_corr_single_sample"
        if feature_mask is None:
            feature_mask = _construct_default_feature_mask(attributions)

        # flatten the feature mask
        feature_mask_flattened, flattened_mask_shape = _tuple_tensors_to_tensors(
            feature_mask
        )

        # validate feature masks are increasing non-negative
        _validate_feature_mask(feature_mask_flattened)

        # remove the batch index
        feature_mask_flattened = feature_mask_flattened.squeeze()

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
        reduced_attributions = reduced_attributions.abs()

        if remove_features_in_ascending_order:
            # get the gathererd-attributions sorted in ascending order of their importance
            # ascending indices can be important as if we remove features in chunks, we want to remove the least important
            # and more important features together instead of mixing them together
            # however this makes it impossible to perturb features together for multi-target scenario
            # overall the monotonicity_corr and non-sensitivity should not change too much for small beam sizes
            # and not at all for beam size 0 as the order of the features is not important in that case
            feature_indices = torch.argsort(reduced_attributions)
        else:
            feature_indices = torch.arange(
                reduced_attributions.size(0), device=inputs[0].device
            )

        # now we generate the global perturbation masks and the chunked reduced attributions
        # each step of the perturbation will generate a perturbation mask for all the features in a single chunk
        # the chunk size is determined by the percentage_feature_removal_per_step
        # if the percentage_feature_removal_per_step is 0.01, then 1% of the features will be removed in each step
        # if the percentage_feature_removal_per_step is 0, then chunk size is set to 1 in which case every
        # perturbation step will remove a single feature in the ascending order of importance
        # the returned chunk_reduced_attributions will have the same shape as the global_perturbation_masks
        # and will contain the sum of attributions over the features in the each chunk
        global_perturbation_masks, chunk_reduced_attributions = (
            _feature_mask_to_chunked_perturbation_mask_with_attributions(
                feature_mask_flattened,
                reduced_attributions,
                feature_indices,
                frozen_features,
                percentage_feature_removal_per_step,
            )
        )
        assert global_perturbation_masks.shape[0] == chunk_reduced_attributions.shape[0]

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

        perturbed_fwd_diffs_relative_vars = np.array(agg_tensors)
        chunk_reduced_attributions = chunk_reduced_attributions.cpu().numpy()
        monotonicity_corr = compute_monotonocity_corr(
            perturbed_fwd_diffs_relative_vars, chunk_reduced_attributions
        )
        non_sens = compute_non_sens(
            perturbed_fwd_diffs_relative_vars, chunk_reduced_attributions
        )
    return (
        monotonicity_corr,
        non_sens,
        n_features,
        perturbed_fwd_diffs_relative_vars,
        chunk_reduced_attributions,
    )


def _monotonicity_corr_and_non_sens(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
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
        # perform argument formattings
        inputs = _format_tensor_into_tuples(inputs)  # type: ignore
        baselines = _format_tensor_into_tuples(baselines)  # type: ignore
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        attributions = _format_tensor_into_tuples(attributions)  # type: ignore
        feature_mask = _format_tensor_into_tuples(feature_mask)  # type: ignore

        # Make sure that inputs and corresponding attributions have matching sizes.
        assert len(inputs) == len(attributions), (
            """The number of tensors in the inputs and
            attributions must match. Found number of tensors in the inputs is: {} and in the
            attributions: {}"""
        ).format(len(inputs), len(attributions))
        if feature_mask is not None:
            for mask, attribution in zip(feature_mask, attributions):
                assert mask.shape == attribution.shape, (
                    """The shape of the feature mask and the attribution
                    must match. Found feature mask shape: {} and attribution shape: {}"""
                ).format(mask.shape, attribution.shape)

        bsz = inputs[0].size(0)
        monotonicity_corr_batch = []
        non_sens_batch = []
        n_features_batch = []
        perturbed_fwd_diffs_relative_vars_batch = []
        feature_group_attribution_scores_batch = []
        for sample_idx in tqdm.tqdm(range(bsz), disable=not show_progress):
            (
                monotonicity_corr,
                non_sens,
                n_features,
                perturbed_fwd_diffs_relative_vars,
                feature_group_attribution_scores,
            ) = eval_monotonicity_corr_and_non_sens_single_sample(
                forward_func=forward_func,
                inputs=tuple(input[sample_idx].unsqueeze(0) for input in inputs),
                attributions=tuple(
                    attr[sample_idx].unsqueeze(0) for attr in attributions
                ),
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
                target=(
                    target[sample_idx]
                    if isinstance(target, (list, torch.Tensor))
                    else target
                ),
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

            monotonicity_corr_batch.append(monotonicity_corr)
            non_sens_batch.append(non_sens)
            n_features_batch.append(n_features)
            perturbed_fwd_diffs_relative_vars_batch.append(
                perturbed_fwd_diffs_relative_vars
            )
            feature_group_attribution_scores_batch.append(
                feature_group_attribution_scores
            )
        monotonicity_corr_batch = torch.tensor(monotonicity_corr_batch)
        non_sens_batch = torch.tensor(non_sens_batch)
        n_features_batch = torch.tensor(n_features_batch)
        perturbed_fwd_diffs_relative_vars_batch = [
            torch.tensor(x) for x in perturbed_fwd_diffs_relative_vars_batch
        ]
        feature_group_attribution_scores_batch = [
            torch.tensor(x) for x in feature_group_attribution_scores_batch
        ]

        return (
            monotonicity_corr_batch,
            non_sens_batch,
            n_features_batch,
            perturbed_fwd_diffs_relative_vars_batch,
            feature_group_attribution_scores_batch,
        )


def monotonicity_corr_and_non_sens(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: Union[
        List[TensorOrTupleOfTensorsGeneric], TensorOrTupleOfTensorsGeneric
    ],
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
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
    return_intermediate_results: bool = False,
    return_dict: bool = False,
    is_multi_target: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Implementation of Monotonicity Correlation and NonSensitivity by Nguyen at el., 2020. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library. Both of these implementations are combined together since the monotonicity and non-sensitivity
    metric are closely related and can be computed together.

    1. Monotonicity measures the (Spearman’s) correlation coefficient of the absolute values of the attributions
    and the uncertainty in probability estimation. The paper argues that if attributions are not monotonic
    then they are not providing the correct importance of the feature.

    2. Non-sensitivity measures if zero-importance is only assigned to features, that the model is not
    functionally dependent on.

    References:
        1) An-phi Nguyen and María Rodríguez Martínez.: "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).
        2) Marco Ancona et al.: "Explaining Deep Neural Networks with a Polynomial Time Algorithm for Shapley
        Values Approximation." ICML (2019): 272-281.
        3) Grégoire Montavon et al.: "Methods for interpreting and
        understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

    Args:
        forward_func (Callable):
                The forward function of the model or any modification of it.

        inputs (Tensor or tuple[Tensor, ...]): Input for which
                attributions are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples (aka batch size), and if
                multiple input tensors are provided, the examples must
                be aligned appropriately.

        baselines (scalar, Tensor, tuple of scalar, or Tensor):
                Baselines define reference values against which the completeness is measured which sometimes
                represent ablated values and are used to compare with the actual inputs to compute
                importance scores in attribution algorithms. They can be represented
                as:

                - a single tensor, if inputs is a single tensor, with
                  exactly the same dimensions as inputs or the first
                  dimension is one and the remaining dimensions match
                  with inputs.

                - a single scalar, if inputs is a single tensor, which will
                  be broadcasted for each input value in input tensor.

                - a tuple of tensors or scalars, the baseline corresponding
                  to each tensor in the inputs' tuple can be:

                - either a tensor with matching dimensions to
                  corresponding tensor in the inputs' tuple
                  or the first dimension is one and the remaining
                  dimensions match with the corresponding
                  input tensor.

                - or a scalar, corresponding to a tensor in the
                  inputs' tuple. This scalar value is broadcasted
                  for corresponding input tensor.

                Default: None

        attributions (Tensor or tuple[Tensor, ...]):
                Attribution scores computed based on an attribution algorithm.
                This attribution scores can be computed using the implementations
                provided in the `captum.attr` package. Some of those attribution
                approaches are so called global methods, which means that
                they factor in model inputs' multiplier, as described in:
                https://arxiv.org/abs/1711.06104
                Many global attribution algorithms can be used in local modes,
                meaning that the inputs multiplier isn't factored in the
                attribution scores.
                This can be done duing the definition of the attribution algorithm
                by passing `multipy_by_inputs=False` flag.
                For example in case of Integrated Gradients (IG) we can obtain
                local attribution scores if we define the constructor of IG as:
                ig = IntegratedGradients(multipy_by_inputs=False)

                Some attribution algorithms are inherently local.
                Examples of inherently local attribution methods include:
                Saliency, Guided GradCam, Guided Backprop and Deconvolution.

                For local attributions we can use real-valued perturbations
                whereas for global attributions that perturbation is binary.
                https://arxiv.org/abs/1901.09392

                If we want to compute the infidelity of global attributions we
                can use a binary perturbation matrix that will allow us to select
                a subset of features from `inputs` or `inputs - baselines` space.
                This will allow us to approximate sensitivity-n for a global
                attribution algorithm.

                `infidelity_perturb_func_decorator` function decorator is a helper
                function that computes perturbations under the hood if perturbed
                inputs are provided.

                For more details about how to use `infidelity_perturb_func_decorator`,
                please, read the documentation about `perturb_func`

                Attributions have the same shape and dimensionality as the inputs.
                If inputs is a single tensor then the attributions is a single
                tensor as well. If inputs is provided as a tuple of tensors
                then attributions will be tuples of tensors as well.

        feature_mask (Tensor or tuple[Tensor, ...], optional):
                    feature_mask defines a mask for the input, grouping
                    features which should be perturbed together. feature_mask
                    should contain the same number of tensors as inputs.
                    Each tensor should
                    be the same size as the corresponding input or
                    broadcastable to match the input tensor. Each tensor
                    should contain integers in the range 0 to num_features
                    - 1, and indices corresponding to the same feature should
                    have the same value.
                    Note that features within each input tensor are perturbed
                    independently (not across tensors).
                    If the forward function returns a single scalar per batch,
                    we enforce that the first dimension of each mask must be 1,
                    since attributions are returned batch-wise rather than per
                    example, so the attributions must correspond to the
                    same features (indices) in each input example.
                    If None, then a feature mask is constructed which assigns
                    each scalar within a tensor as a separate feature, which
                    is perturbed independently.
                    Default: None

        additional_forward_args (Any, optional): If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors
                or any arbitrary python types. These arguments are provided to
                forward_func in order, following the arguments in inputs.
                Note that the perturbations are not computed with respect
                to these arguments. This means that these arguments aren't
                being passed to `perturb_func` as an input argument.

                Default: None
        target (int, tuple, Tensor, or list, optional): Indices for selecting
                predictions from output(for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example, no target
                index is necessary.
                For general 2D outputs, targets can be either:

                - A single integer or a tensor containing a single
                  integer, which is applied to all input examples

                - A list of integers or a 1D tensor, with length matching
                  the number of examples in inputs (dim 0). Each integer
                  is applied as the target for the corresponding example.

                  For outputs with > 2 dimensions, targets can be either:

                - A single tuple, which contains #output_dims - 1
                  elements. This target index is applied to all examples.

                - A list of tuples with length equal to the number of
                  examples in inputs (dim 0), and each tuple containing
                  #output_dims - 1 elements. Each tuple is applied as the
                  target for the corresponding example.

                Default: None

        perturb_func (Callable):
                The perturbation function of model inputs. This function takes
                model inputs and the corresponding feature masks to be perturbed.
                Optionally it also takes baselines as input arguments and returns
                a tuple of perturbed inputs. For example:

                >>> def my_perturb_func(inputs, masks, baselines):
                >>>   <MY-LOGIC-HERE>
                >>>   return perturbed_inputs

                If there are more than one inputs passed to this function those
                will be passed to `perturb_func` as tuples in the same order as they
                are passed to the function. In addition the corresponding feature masks
                will also be passed as a tuple in the same order as inputs. See default_perturb_func
                in metrics._utils.perturbation.py for an example of a perturbation function.

                If inputs
                 - is a single tensor, the function needs to return single tensor of perturbed inputs.
                 - is a tuple of tensors, corresponding perturbed
                   inputs must be computed and returned as tuples in the
                   following format:

                   (perturbed_input1, perturbed_input2, ... perturbed_inputN)

                It is important to note that for performance reasons `perturb_func`
                isn't called for each example individually but on a batch of
                input examples that are repeated `max_features_processed_per_batch / batch_size`
                times within the batch.

        n_perturbations_per_feature (int, optional): The number of times each feature is perturbed.
                Each input example in the inputs tensor is expanded `n_perturbations_per_feature`
                times before calling `perturb_func` function for every single feature in the input.
                So if you have an input tensor of shape (N, C, H, W) and you set `n_perturbations_per_feature`,
                then each example in the batch will be handled separately and for each example a total of
                `C * H * W` perturbation steps will be performed and in each perturbation step a single feature
                will be repeated `n_perturbations_per_feature` times. This means that the total number of
                perturbation steps will be `C * H * W * n_perturbations_per_feature` for each example in the batch.

                Default: 10
        max_features_processed_per_batch (int, optional): The number of maximum input
                features that are processed together for every example. In case the number of
                features in each example (`C * H * W`) exceeds
                `max_features_processed_per_batch`, they will be sliced
                into batches of `max_features_processed_per_batch` examples and processed
                in a sequential order. However the total effective batch size will still be
                `max_features_processed_per_batch * n_perturbations_per_feature` as in each
                perturbation step, `max_features_processed_per_batch * n_perturbations_per_feature` features
                are processed. If `max_features_processed_per_batch` is None, all
                examples are processed together. `max_features_processed_per_batch` should
                at least be equal `n_perturbations_per_feature` and at most
                `C * H * W * n_perturbations_per_feature`.
        frozen_features (List[torch.Tensor], optional): A list of frozen features that are not perturbed.
                This can be useful for ignoring the input structure features like padding, etc. Default: None
                In case CLS,PAD,SEP tokens are present in the input, they can be frozen by passing the indices
                of feature masks that correspond to these tokens.
        zero_attribution_threshold (float, optional): Defines the minimum threshold for the attribution scores and the model forward
                variances. If the absolute value of the attribution scores or the model forward variances
                is less than `zero_attribution_threshold`, it is considered as zero. This is used to compute the non-sensitivity
                metric. Default: 1e-5

        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class.
                Default is False.

        show_progress (bool, optional): Displays the progress of the computation. Default: True
        return_intermediate_results (bool, optional): A boolean flag that indicates whether the intermediate results
                of the metric computation are returned.
                Default is False.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
                with keys as the metric names and values as the corresponding metric outputs.
                Default is False.
    Returns:
        A tuple of tensors:
            monotonicity_corr_batch (Tensor): A tensor of scalar monotonicity_corr scores per
                    input example. The first dimension is equal to the
                    number of examples in the input batch and the second
                    dimension is one.
            non_sensitivity_batch (Tensor): A tensor of scalar non_sensitivity scores per
                    input example. The first dimension is equal to the
                    number of examples in the input batch and the second
                    dimension is one.
            n_features_batch (Tensor): A tensor of scalar values that represent the total number of
                    features that are processed in the input batch. The first dimension is equal to the
                    number of examples in the input batch and the second dimension is one.

    Examples::
        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> saliency = Saliency(net)
        >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> baselines = torch.zeros(2, 3, 32, 32)
        >>> # Computes saliency maps for class 3.
        >>> attribution = saliency.attribute(input, target=3)
        >>> # define a perturbation function for the input

        >>> # Computes the monotonicity correlation and non-sensitivity scores for saliency maps
        >>> monotonicity_corr, non_sens, n_features = monotonicity_corr_and_non_sens(net, input, attribution, baselines)
    """
    metric_func = (
        _multi_target_monotonicity_corr_and_non_sens
        if is_multi_target
        else _monotonicity_corr_and_non_sens
    )
    (
        monotonicity_corr_batch,
        non_sens_batch,
        n_features_batch,
        perturbed_fwd_diffs_relative_vars_batch,
        feature_group_attribution_scores_batch,
    ) = metric_func(
        forward_func=forward_func,
        inputs=inputs,
        **(
            dict(attributions_list=attributions)
            if is_multi_target
            else dict(attributions=attributions)
        ),
        baselines=baselines,
        feature_mask=feature_mask,
        additional_forward_args=additional_forward_args,
        **(dict(targets_list=target) if is_multi_target else dict(target=target)),
        frozen_features=frozen_features,
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
    if return_intermediate_results:
        if return_dict:
            return {
                "monotonicity_corr_score": monotonicity_corr_batch,
                "non_sensitivity_score": non_sens_batch,
                "n_features": n_features_batch,
                "perturbed_fwd_diffs_relative_vars": perturbed_fwd_diffs_relative_vars_batch,
                "feature_group_attribution_scores": feature_group_attribution_scores_batch,
            }
        else:
            return (
                monotonicity_corr_batch,
                non_sens_batch,
                n_features_batch,
                perturbed_fwd_diffs_relative_vars_batch,
                feature_group_attribution_scores_batch,
            )
    else:
        if return_dict:
            return {
                "monotonicity_corr_score": monotonicity_corr_batch,
                "non_sensitivity_score": non_sens_batch,
            }
        else:
            return monotonicity_corr_batch, non_sens_batch
