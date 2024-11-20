#!/usr/bin/env python3

from typing import Any, Callable, Optional, Tuple, Union, cast

import numpy as np
import torch
import tqdm
from captum._utils.common import (
    ExpansionTypes,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    _run_forward,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor

from torchxai.metrics._utils.batching import _divide_and_aggregate_metrics_n_features
from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _reduce_tensor_with_indices,
    _split_tensors_to_tuple_tensors,
    _tuple_tensors_to_tensors,
    _validate_feature_mask,
)


def eval_monotonicity_single_sample_tupled_computation(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_features_processed_per_batch: int = None,
    frozen_features: Optional[torch.Tensor] = None,
    show_progress: bool = False,
) -> Tensor:
    def _next_monotonicity_tensors(
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
            for mask in feature_mask
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

    def _agg_monotonicity_tensors(agg_tensors, tensors):
        return tuple(agg_t + t for agg_t, t in zip(agg_tensors, tensors))

    bsz = inputs[0].size(0)
    assert bsz == 1, "Batch size must be 1 for monotonicity_single_sample"
    if feature_mask is not None:
        # assert that all elements in the feature_mask are unique and non-negative increasing
        _validate_feature_mask(feature_mask)
    else:
        feature_mask = _construct_default_feature_mask(attributions)

    # this assumes a batch size of 1, this will not work for batch size > 1
    n_features = max(x.max() for x in feature_mask).item() + 1

    # gather attribution scores of feature groups
    # this can be useful for efficiently summing up attributions of feature groups
    gathered_attributions = tuple()
    for attribution, mask in zip(attributions, feature_mask):
        gathered_attribution = torch.zeros_like(attribution)
        reduced_indices = mask.squeeze() - mask.min()
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
            _next_monotonicity_tensors,
            agg_func=_agg_monotonicity_tensors,
            max_features_processed_per_batch=max_features_processed_per_batch,
        )

        # compute monotonicity corr metric
        def compute_monotonicity(
            baseline_perturbed_fwds: np.ndarray,
        ):
            # as feature are added from least to higher importance, the forward outputs should be monotonically increasing
            return np.all(np.diff(baseline_perturbed_fwds) >= 0)

        agg_tensors = tuple(np.array(x) for x in agg_tensors)
        baseline_perturbed_fwds = agg_tensors[0]
        monotonicity = compute_monotonicity(baseline_perturbed_fwds)
    return monotonicity


def monotonicity_tupled_computation(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_features_processed_per_batch: int = None,
) -> Tensor:
    # perform argument formattings
    inputs = _format_tensor_into_tuples(inputs)  # type: ignore
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    attributions = _format_tensor_into_tuples(attributions)  # type: ignore
    feature_mask = _format_tensor_into_tuples(feature_mask)  # type: ignore

    # Make sure that inputs and corresponding attributions have matching sizes.
    assert len(inputs) == len(attributions), (
        """The number of tensors in the inputs and
        attributions must match. Found number of tensors in the inputs is: {} and in the
        attributions: {}"""
    ).format(len(inputs), len(attributions))

    bsz = inputs[0].size(0)
    monotonicity_batch = []
    for sample_idx in range(bsz):
        monotonicity = eval_monotonicity_single_sample_tupled_computation(
            forward_func=forward_func,
            inputs=tuple(input[sample_idx].unsqueeze(0) for input in inputs),
            attributions=tuple(attr[sample_idx].unsqueeze(0) for attr in attributions),
            feature_mask=(
                tuple(mask[sample_idx].unsqueeze(0) for mask in feature_mask)
                if feature_mask is not None
                else None
            ),
            baselines=tuple(
                baseline[sample_idx].unsqueeze(0) for baseline in baselines
            ),
            additional_forward_args=(
                x[sample_idx].unsqueeze(0) for x in additional_forward_args
            ),
            target=target[sample_idx],
            max_features_processed_per_batch=max_features_processed_per_batch,
        )
        monotonicity_batch.append(monotonicity)
    monotonicity_batch = torch.tensor(monotonicity_batch)
    return monotonicity_batch


def eval_monotonicity_single_sample(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_features_processed_per_batch: int = None,
    frozen_features: Optional[torch.Tensor] = None,
    show_progress: bool = False,
) -> Tensor:
    def _next_monotonicity_tensors(
        current_n_perturbed_features: int,
        current_n_steps: int,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]:
        # get the indices of features that will be perturbed in the current iteration
        # for example if we do 1 by 1 perturbation, then the first iteration will perturb the first feature
        # the second iteration will perturb both the first and second feature and so on
        baselines_perturbed = []
        for feature_idx in range(
            current_n_steps - current_n_perturbed_features, current_n_steps
        ):
            # for each feature in the current step incrementally replace the baseline with the original sample
            if (
                frozen_features is not None
                and ascending_attribution_indices[feature_idx] in frozen_features
            ):
                # freeze this feature
                perturbation_mask = torch.zeros_like(
                    feature_mask, device=inputs.device
                ).bool()
            else:
                perturbation_mask = (
                    feature_mask == ascending_attribution_indices[feature_idx]
                )
            if len(inputs.shape) > len(perturbation_mask.shape):
                perturbation_mask = perturbation_mask.unsqueeze(-1)
            global_perturbed_baselines[perturbation_mask.expand_as(inputs)] = inputs[
                perturbation_mask.expand_as(inputs)
            ]  # input[0] here since batch size is 1
            baselines_perturbed.append(
                global_perturbed_baselines[0].clone()
            )  # input[0] here since batch size is 1
        baselines_perturbed = torch.stack(baselines_perturbed)

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
            # the inputs are [batch_size, feature_size, feature_dims] so we need to split them by feature size
            _split_tensors_to_tuple_tensors(
                baselines_perturbed, baselines_shape, dim=1
            ),
            targets_expanded,
            additional_forward_args_expanded,
        )

        # this is a list of tensors which holds the forward outputs of the model
        # on each feature group perturbation
        # the first element will be when the feature with lowest importance is added to the baseline
        # the last element will be when all features are added to the baseline
        return (list(baselines_perturbed_fwd.detach().cpu().numpy()),)

    with torch.no_grad():

        def _agg_monotonicity_tensors(agg_tensors, tensors):
            return tuple(agg_t + t for agg_t, t in zip(agg_tensors, tensors))

        bsz = inputs[0].size(0)
        assert bsz == 1, "Batch size must be 1 for monotonicity_single_sample"

        # flatten all inputs and baseline features in the input
        inputs, inputs_shape = _tuple_tensors_to_tensors(inputs)
        global_perturbed_baselines, baselines_shape = _tuple_tensors_to_tensors(
            baselines
        )
        assert (
            inputs_shape == baselines_shape
        ), "Inputs and baselines must have the same shape"

        # flatten all feature masks in the input
        if feature_mask is not None:
            feature_mask, _ = _tuple_tensors_to_tensors(feature_mask)
        else:
            feature_mask = _construct_default_feature_mask(attributions)
            feature_mask, _ = _tuple_tensors_to_tensors(feature_mask)

        # flatten all attributions in the input, this must be done after the feature masks are flattened as
        # feature masks may depened on attribution
        attributions, _ = _tuple_tensors_to_tensors(attributions)

        # validate feature masks are increasing non-negative
        _validate_feature_mask(feature_mask)

        # gather attribution scores of feature groups
        # this can be useful for efficiently summing up attributions of feature groups
        # this is why we need a single batch size as gathered attributes and number of features for each
        # sample can be different
        reduced_attributions, n_features = _reduce_tensor_with_indices(
            attributions[0], indices=feature_mask[0].flatten()
        )

        # get the gathererd-attributions sorted in ascending order of their importance
        ascending_attribution_indices = torch.argsort(reduced_attributions)

        # the logic for this implementation as as follows:
        # we start from baseline and in each iteration, a feature group is replaced by the original sample
        # in ascending order of its importance
        agg_tensors = _divide_and_aggregate_metrics_n_features(
            n_features,
            _next_monotonicity_tensors,
            agg_func=_agg_monotonicity_tensors,
            max_features_processed_per_batch=max_features_processed_per_batch,
            show_progress=show_progress,
        )

        # compute monotonicity corr metric
        def compute_monotonicity(
            baseline_perturbed_fwds: np.ndarray,
        ):
            # as feature are added from least to higher importance, the forward outputs should be monotonically increasing
            return np.all(np.diff(baseline_perturbed_fwds) >= 0)

        agg_tensors = tuple(np.array(x).flatten() for x in agg_tensors)
        baseline_perturbed_fwds = agg_tensors[0]
        monotonicity_score = compute_monotonicity(baseline_perturbed_fwds)
    return monotonicity_score, torch.from_numpy(baseline_perturbed_fwds)


def monotonicity(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_features_processed_per_batch: Optional[int] = None,
    is_multi_target: bool = False,
    show_progress: bool = False,
    return_intermediate_results: bool = False,
    return_dict: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Implementation of Monotonicity metric by Arya at el., 2019. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library.

    Monotonicity tests if adding more positive evidence increases the probability
    of classification in the specified class.

    It captures attributions' faithfulness by incrementally adding each attribute
    in order of increasing importance and evaluating the effect on model performance.
    As more features are added, the performance of the model is expected to increase
    and thus result in monotonically increasing model performance.

    References:
        1) Vijay Arya et al.: "One explanation does not fit all: A toolkit and taxonomy of ai explainability
        techniques." arXiv preprint arXiv:1909.03012 (2019).
        2) Ronny Luss et al.: "Generating contrastive explanations with monotonic attribute functions."
        arXiv preprint arXiv:1905.12698 (2019).

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
                to these arguments.

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
        max_features_processed_per_batch (int, optional): The number of maximum input
                features that are processed together for every example. In case the number of
                features to be perturbed in each example (`total_features_perturbed`) exceeds
                `max_features_processed_per_batch`, they will be sliced
                into batches of `max_features_processed_per_batch` examples and processed
                in a sequential order.
        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class.
                Default is False.
        show_progress (bool, optional): Displays the progress of computation.
        return_intermediate_results (bool, optional): A boolean flag that indicates whether the intermediate
                results of the metric computation are returned. If set to True, the intermediate results
                are returned as a tuple of tensors. Default is False.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
                with keys as the metric names and values as the corresponding metric outputs.
                Default is False.
    Returns:
        Returns:
            A tuple of three tensors:
            Tensor: - The monotonicity scores of the batch. The first dimension is equal to the
                    number of examples in the input batch and the second dimension is 1.
            Tensor: - The forward outputs when features are slowly added to baseline for monotonicity computation.
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

        >>> # Computes the monotonicity scores for saliency maps
        >>> monotonicity, asc_baseline_perturb_fwds = monotonicity(net, input, attribution, baselines)
    """

    if is_multi_target:
        # monotonicity cannot be batched across targets, this is because it requires the removal
        # of attribution features for each target in an desencding order of their importance. Since for each target
        # the order of importance of features can be different, we cannot batch the computation across targets.
        attributions_list = attributions
        targets_list = target
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

        monotonicity_batch_list = []
        asc_baseline_perturbed_fwds_batch_list = []
        for attributions, target in zip(attributions_list, targets_list):
            (monotonicity_batch, asc_baseline_perturbed_fwds_batch) = monotonicity(
                forward_func=forward_func,
                inputs=inputs,
                attributions=attributions,
                baselines=baselines,
                feature_mask=feature_mask,
                additional_forward_args=additional_forward_args,
                target=target,
                max_features_processed_per_batch=max_features_processed_per_batch,
                show_progress=show_progress,
                return_dict=False,
                return_intermediate_results=True,
            )
            monotonicity_batch_list.append(monotonicity_batch)
            asc_baseline_perturbed_fwds_batch_list.append(
                asc_baseline_perturbed_fwds_batch
            )

        if return_intermediate_results:
            if return_dict:
                {
                    "monotonicity_score": monotonicity_batch_list,
                    "asc_baseline_perturbed_fwds_batch": asc_baseline_perturbed_fwds_batch_list,
                }
            else:
                return monotonicity_batch_list, asc_baseline_perturbed_fwds_batch_list
        else:
            if return_dict:
                return {"monotonicity_score": monotonicity_batch_list}
            return monotonicity_batch_list

    with torch.no_grad():
        # perform argument formattings
        inputs = _format_tensor_into_tuples(inputs)  # type: ignore
        if baselines is None:
            baselines = tuple(torch.zeros_like(inp) for inp in inputs)
        else:
            baselines = _format_baseline(baselines, cast(Tuple[Tensor, ...], inputs))
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
            assert len(feature_mask) == len(feature_mask), (
                """The number of tensors in the inputs and
                feature_masks must match. Found number of tensors in the inputs is: {} and in the
                attributions: {}"""
            ).format(len(feature_mask), len(feature_mask))
            for input, attribution, mask in zip(inputs, attributions, feature_mask):
                assert input.shape == mask.shape == attribution.shape, (
                    """
                    The shape of the input, attribution and feature mask must match. Found shapes are: input {}
                    attribution {} and feature mask {}
                    """
                ).format(input.shape, attribution.shape, mask.shape)

        bsz = inputs[0].size(0)
        monotonicity_batch = []
        asc_baseline_perturbed_fwds_batch = []
        for sample_idx in tqdm.tqdm(range(bsz), disable=not show_progress):
            monotonicity_score, asc_baseline_perturbed_fwds = (
                eval_monotonicity_single_sample(
                    forward_func=forward_func,
                    inputs=tuple(input[sample_idx].unsqueeze(0) for input in inputs),
                    attributions=tuple(
                        attr[sample_idx].unsqueeze(0) for attr in attributions
                    ),
                    feature_mask=(
                        tuple(mask[sample_idx].unsqueeze(0) for mask in feature_mask)
                        if feature_mask is not None
                        else None
                    ),
                    baselines=(
                        tuple(
                            baseline[sample_idx].unsqueeze(0) for baseline in baselines
                        )
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
                    target=(
                        target[sample_idx]
                        if isinstance(target, (list, torch.Tensor))
                        else target
                    ),
                    max_features_processed_per_batch=max_features_processed_per_batch,
                    show_progress=show_progress,
                )
            )
            monotonicity_batch.append(monotonicity_score)
            asc_baseline_perturbed_fwds_batch.append(asc_baseline_perturbed_fwds)
        monotonicity_batch = torch.tensor(monotonicity_batch)

        if return_intermediate_results:
            if return_dict:
                {
                    "monotonicity_score": monotonicity_batch,
                    "asc_baseline_perturbed_fwds_batch": asc_baseline_perturbed_fwds_batch,
                }
            else:
                return monotonicity_batch, asc_baseline_perturbed_fwds_batch
        else:
            if return_dict:
                return {"monotonicity_score": monotonicity_batch}
            return monotonicity_batch
