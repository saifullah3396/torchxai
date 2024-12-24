from collections import OrderedDict
from typing import Any, Callable, List, Optional, Tuple, Union, cast

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
from torchxai.metrics._utils.batching import (
    _divide_and_aggregate_metrics_n_perturbations_per_feature,
)
from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _feature_mask_to_chunked_accumulated_perturbation_mask,
    _format_tensor_feature_dim,
    _reduce_tensor_with_indices_non_deterministic,
    _split_tensors_to_tuple_tensors,
    _tuple_tensors_to_tensors,
    _validate_feature_mask,
)


def compute_aopc_scores(
    inputs_perturbed_fwds: torch.Tensor, input_fwds: torch.Tensor
) -> torch.Tensor:
    """
    Computes the AOPC score for the given input perturbations and forward outputs for a single sample.

    Args:
        inputs_perturbed_fwds (torch.Tensor): The forward outputs of the model on the perturbed inputs. The shape of
            the tensor is [(2*n_random_perms), n_features]. The first row corresponds to the descending order of feature
            importance, the second row corresponds to the ascending order of feature importance and the rest of the rows
            correspond to the random order of feature importance.
        input_fwds (torch.Tensor): The forward output of the model on the original input.
    """

    # concatenate the input forward output with the perturbed input forward outputs
    cat_fwds = torch.cat(
        [input_fwds.repeat(inputs_perturbed_fwds.shape[0], 1), inputs_perturbed_fwds],
        dim=1,
    )

    # get aopc scores
    aopc_scores_batch = []
    for fwd_scores in cat_fwds:
        cumulative_value = 0.0
        aopc_scores = []
        for n, curr_score in enumerate(fwd_scores):
            cumulative_value += input_fwds - curr_score
            aopc_scores.append(cumulative_value / (n + 1))
        aopc_scores_batch.append(aopc_scores)
    aopc_scores_batch = torch.tensor(aopc_scores_batch)
    return aopc_scores_batch


def compute_aopc_scores_vectorized(
    inputs_perturbed_fwds: torch.Tensor, input_fwds: torch.Tensor
) -> torch.Tensor:
    """
    Computes the AOPC score in a vectorized manner for the given input perturbations and forward outputs
    for a single sample.

    Args:
        inputs_perturbed_fwds (torch.Tensor): The forward outputs of the model on the perturbed inputs. The shape of
            the tensor is [(2*n_random_perms), n_features]. The first row corresponds to the descending order of feature
            importance, the second row corresponds to the ascending order of feature importance and the rest of the rows
            correspond to the random order of feature importance.
        input_fwds (torch.Tensor): The forward output of the model on the original input.
    """

    # concatenate the input forward output with the perturbed input forward outputs
    input_fwds = _format_tensor_feature_dim(input_fwds)
    cat_fwds = torch.cat(
        [input_fwds.repeat(inputs_perturbed_fwds.shape[0], 1), inputs_perturbed_fwds],
        dim=1,
    )

    # Convert input_fwds to tensor for broadcasting
    input_fwds_tensor = input_fwds.expand_as(cat_fwds)

    # Compute the differences between input_fwds and each score
    differences = input_fwds_tensor - cat_fwds

    # Compute cumulative sum along the rows (axis=1)
    cumulative_sums = torch.cumsum(differences, dim=1)

    # Compute the number of elements considered so far
    counts = torch.arange(
        1, inputs_perturbed_fwds.shape[1] + 2, device=cumulative_sums.device
    ).float()
    counts = counts.expand_as(cumulative_sums)

    # Compute AOPC scores
    aopc_scores = cumulative_sums / counts

    return aopc_scores


def eval_aopcs_single_sample(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_features_processed_per_batch: Optional[int] = None,
    total_feature_bins: int = 100,
    frozen_features: Optional[List[torch.Tensor]] = None,
    n_random_perms: int = 10,
    seed: Optional[int] = None,
    show_progress: bool = False,
) -> Tensor:
    def _generate_perturbations(
        current_n_perturbed_features: int,
        current_perturbation_mask: TensorOrTupleOfTensorsGeneric,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
        # repeat each current_n_perturbed_features times
        inputs_expanded = tuple(
            input.repeat(
                current_n_perturbed_features * (2 + n_random_perms),
                *tuple([1] * len(input.shape[1:])),
            )
            for input in inputs
        )

        # split back to tuple tensors
        perturbation_mask_expanded = _split_tensors_to_tuple_tensors(
            current_perturbation_mask, flattened_mask_shape
        )

        # view as input shape (this is only necessary for edge cases where input is of (1, 1) shape)
        perturbation_mask_expanded = tuple(
            mask.view_as(input)
            for mask, input in zip(perturbation_mask_expanded, inputs_expanded)
        )
        return tuple(
            input * ~mask + baseline * mask
            for input, mask, baseline in zip(
                inputs_expanded, perturbation_mask_expanded, baselines
            )
        )

    def _generate_baseline_perturbations(
        current_n_perturbed_features: int,
        current_perturbation_mask: TensorOrTupleOfTensorsGeneric,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
        # repeat each current_n_perturbed_features times
        inputs_expanded = tuple(
            input.repeat(
                current_n_perturbed_features * (2 + n_random_perms),
                *tuple([1] * len(input.shape[1:])),
            )
            for input in inputs
        )
        # repeat each current_n_perturbed_features times
        baselines_expanded = tuple(
            baseline.repeat(
                current_n_perturbed_features * (2 + n_random_perms),
                *tuple([1] * len(baseline.shape[1:])),
            )
            for baseline in baselines
        )

        # split back to tuple tensors
        perturbation_mask_expanded = _split_tensors_to_tuple_tensors(
            current_perturbation_mask, flattened_mask_shape
        )

        # view as input shape (this is only necessary for edge cases where input is of (1, 1) shape)
        perturbation_mask_expanded = tuple(
            mask.view_as(input)
            for mask, input in zip(perturbation_mask_expanded, inputs_expanded)
        )
        return tuple(
            baseline * ~mask + input * mask
            for baseline, mask, input in zip(
                baselines_expanded, perturbation_mask_expanded, inputs_expanded
            )
        )

    def _next_aopc_tensors(
        current_n_perturbed_features: int,
        current_n_steps: int,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]:
        # get the indices of features that will be perturbed in the current iteration
        # for example if we do 1 by 1 perturbation, then the first iteration will perturb the first feature
        # the second iteration will perturb both the first and second feature and so on
        current_feature_indices = torch.arange(
            current_n_steps - current_n_perturbed_features,
            current_n_steps,
            device=inputs[0].device,
        )

        inputs_perturbed = {}
        curr_perturbation_mask = []
        for (
            key,
            global_perturbation_masks,
        ) in global_perturbation_masks_per_order.items():
            if key == "rand":
                curr_perturbation_mask.append(
                    global_perturbation_masks[:, current_feature_indices, :]
                )
            else:
                curr_perturbation_mask.append(
                    global_perturbation_masks[current_feature_indices].unsqueeze(0)
                )

        # reshape from desc desc..., asc asc..., rand rand..., to desc asc rand, desc asc rand
        curr_perturbation_mask = torch.cat(
            [x.transpose(0, 1) for x in curr_perturbation_mask], dim=1
        )

        # flatten the perturbation mask to batch dim
        curr_perturbation_mask = curr_perturbation_mask.view(
            -1, curr_perturbation_mask.shape[-1]
        )
        inputs_perturbed = _generate_perturbations(
            current_n_perturbed_features,
            curr_perturbation_mask,
        )
        baselines_perturbed = _generate_baseline_perturbations(
            current_n_perturbed_features,
            curr_perturbation_mask,
        )
        targets_expanded = _expand_target(
            target,
            current_n_perturbed_features * (2 + n_random_perms),
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        additional_forward_args_expanded = _expand_additional_forward_args(
            additional_forward_args,
            current_n_perturbed_features * (2 + n_random_perms),
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        inputs_perturbed_fwd = _run_forward(
            forward_func,
            inputs_perturbed,
            targets_expanded,
            additional_forward_args_expanded,
        )
        baselines_perturbed_fwd = _run_forward(
            forward_func,
            baselines_perturbed,
            targets_expanded,
            additional_forward_args_expanded,
        )

        # we expect a single output per batch
        inputs_perturbed_fwd = inputs_perturbed_fwd.squeeze(-1)
        baselines_perturbed_fwd = baselines_perturbed_fwd.squeeze(-1)

        # reshape outputs to [desc, asc, rand * n_random_perms]
        inputs_perturbed_fwd = torch.stack(
            inputs_perturbed_fwd.split(2 + n_random_perms), dim=0
        )
        baselines_perturbed_fwd = torch.stack(
            baselines_perturbed_fwd.split(2 + n_random_perms), dim=0
        )

        # this is a list of tensors which holds the forward outputs of the model
        # on each feature group perturbation
        # the first element will be when the feature with lowest importance is added to the baseline
        # the last element will be when all features are added to the baseline
        return inputs_perturbed_fwd, baselines_perturbed_fwd

    def _cat_aopc_tensors(agg_tensors, tensors):
        return tuple(torch.cat([x, y], dim=0) for x, y in zip(agg_tensors, tensors))

    with torch.no_grad():
        bsz = inputs[0].size(0)
        assert bsz == 1, "Batch size must be 1 for aopc_single_sample"

        # get the first input forward output
        inputs_fwd = _run_forward(forward_func, inputs, target, additional_forward_args)

        # flatten all feature masks in the input
        if feature_mask is None:
            feature_mask = _construct_default_feature_mask(attributions)

        # flatten all feature masks in the input
        feature_mask_flattened, flattened_mask_shape = _tuple_tensors_to_tensors(
            feature_mask
        )

        # validate feature masks are increasing non-negative
        _validate_feature_mask(feature_mask_flattened)

        # squeeze feature_mask_flattened
        feature_mask_flattened = feature_mask_flattened.squeeze()

        # flatten all attributions in the input, this must be done after the feature masks are flattened as
        # feature masks may depened on attribution
        attributions, _ = _tuple_tensors_to_tensors(attributions)

        # gather attribution scores of feature groups
        # this can be useful for efficiently summing up attributions of feature groups
        # this is why we need a single batch size as gathered attributes and number of features for each
        # sample can be different
        reduced_attributions, n_features = (
            _reduce_tensor_with_indices_non_deterministic(
                attributions[0], indices=feature_mask_flattened
            )
        )

        attribution_indices = OrderedDict()

        # get the gathererd-attributions sorted in descending order of their importance
        attribution_indices["desc"] = torch.argsort(
            reduced_attributions, descending=True
        )

        # get the gathererd-attributions sorted in ascending order of their importance
        attribution_indices["asc"] = torch.argsort(reduced_attributions)

        # get the gathererd-attributions sorted in a random order of their importance n times
        generator = (
            torch.Generator(device=inputs[0].device).manual_seed(seed)
            if seed is not None
            else None
        )
        attribution_indices["rand"] = torch.stack(
            [
                torch.randperm(
                    len(reduced_attributions),
                    generator=generator,
                    device=inputs[0].device,
                )
                for _ in range(n_random_perms)
            ]
        )

        assert attribution_indices["desc"].max() < n_features
        assert attribution_indices["asc"].max() < n_features
        assert attribution_indices["rand"].max() < n_features

        global_perturbation_masks_per_order = OrderedDict()
        for key, indices in attribution_indices.items():
            if key == "rand":
                global_perturbation_masks_per_order[key] = torch.stack(
                    [
                        _feature_mask_to_chunked_accumulated_perturbation_mask(
                            feature_mask_flattened,
                            indices[rand_perm_idx],
                            frozen_features,
                            n_percentage_features_per_step=1 / total_feature_bins,
                        )
                        for rand_perm_idx in range(n_random_perms)
                    ],
                )
            else:
                global_perturbation_masks_per_order[key] = (
                    _feature_mask_to_chunked_accumulated_perturbation_mask(
                        feature_mask_flattened,
                        indices,
                        frozen_features,
                        n_percentage_features_per_step=1 / total_feature_bins,
                    )
                )

        # features are updated after frozen features may have been removed
        n_features = global_perturbation_masks_per_order["desc"].shape[0]

        # the logic for this implementation as as follows:
        # we start from baseline and in each iteration, a feature group is replaced by the original sample
        # in ascending order of its importance
        inputs_perturbed_fwds_agg, baselines_perturbed_fwds_agg = (
            _divide_and_aggregate_metrics_n_perturbations_per_feature(
                n_perturbations_per_feature=(
                    2 + n_random_perms
                ),  # 2 for ascending and descending and n_random_perms for random
                n_features=min(total_feature_bins, n_features),
                metric_func=_next_aopc_tensors,
                agg_func=_cat_aopc_tensors,
                max_features_processed_per_batch=max_features_processed_per_batch,
                show_progress=show_progress,
            )
        )
        inputs_perturbed_aopc_scores = compute_aopc_scores_vectorized(
            inputs_perturbed_fwds_agg.T, inputs_fwd
        )
        baselines_perturbed_aopc_scores = compute_aopc_scores_vectorized(
            baselines_perturbed_fwds_agg.T, baselines_perturbed_fwds_agg.T[0][0]
        )

    return (
        inputs_perturbed_aopc_scores,
        inputs_perturbed_fwds_agg,
        baselines_perturbed_aopc_scores,
        baselines_perturbed_fwds_agg,
        inputs_fwd,
    )


def _aopc(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_features_processed_per_batch: Optional[int] = None,
    total_feature_bins: int = 100,
    frozen_features: Optional[List[torch.Tensor]] = None,
    n_random_perms: int = 10,
    seed: Optional[int] = None,
    show_progress: bool = False,
) -> Any:
    # perform argument formattings
    inputs = _format_tensor_into_tuples(inputs)  # type: ignore
    if baselines is None:
        baselines = tuple(torch.zeros_like(inp) for inp in inputs)
    else:
        baselines = _format_baseline(baselines, cast(Tuple[Tensor, ...], inputs))
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
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
    inputs_perturbed_aopc_scores_batch = []
    inputs_perturbed_fwds_agg_batch = []
    inputs_fwd_batch = []
    baselines_perturbed_aopc_scores_batch = []
    baselines_perturbed_fwds_agg_batch = []
    for sample_idx in tqdm.tqdm(range(bsz), disable=not show_progress):
        (
            inputs_perturbed_aopc_scores,
            inputs_perturbed_fwds_agg,
            baselines_perturbed_aopc_scores,
            baselines_perturbed_fwds_agg,
            inputs_fwd,
        ) = eval_aopcs_single_sample(
            forward_func=forward_func,
            inputs=tuple(input[sample_idx].unsqueeze(0) for input in inputs),
            attributions=tuple(attr[sample_idx].unsqueeze(0) for attr in attributions),
            feature_mask=(
                tuple(mask[sample_idx].unsqueeze(0) for mask in feature_mask)
                if feature_mask is not None
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
            target=(
                target[sample_idx]
                if isinstance(target, (list, torch.Tensor))
                else target
            ),
            max_features_processed_per_batch=max_features_processed_per_batch,
            frozen_features=(
                frozen_features[sample_idx]
                if frozen_features is not None
                else frozen_features
            ),
            total_feature_bins=total_feature_bins,
            n_random_perms=n_random_perms,
            seed=seed,
            show_progress=show_progress,
        )
        inputs_perturbed_aopc_scores_batch.append(inputs_perturbed_aopc_scores)
        inputs_perturbed_fwds_agg_batch.append(inputs_perturbed_fwds_agg)
        baselines_perturbed_aopc_scores_batch.append(baselines_perturbed_aopc_scores)
        baselines_perturbed_fwds_agg_batch.append(baselines_perturbed_fwds_agg)
        inputs_fwd_batch.append(inputs_fwd)

    def _convert_to_tensor_if_possible(list_of_tensors):
        if all([x.shape == list_of_tensors[0].shape for x in list_of_tensors]):
            return torch.stack(list_of_tensors)
        return list_of_tensors

    return (
        _convert_to_tensor_if_possible(
            [x[0] for x in inputs_perturbed_aopc_scores_batch]
        ),
        _convert_to_tensor_if_possible(
            [x[1] for x in inputs_perturbed_aopc_scores_batch]
        ),
        _convert_to_tensor_if_possible(
            [x[2:] for x in inputs_perturbed_aopc_scores_batch]
        ),
        _convert_to_tensor_if_possible(
            [x[0] for x in baselines_perturbed_aopc_scores_batch]
        ),
        _convert_to_tensor_if_possible(
            [x[1] for x in baselines_perturbed_aopc_scores_batch]
        ),
        _convert_to_tensor_if_possible(
            [x[2:] for x in baselines_perturbed_aopc_scores_batch]
        ),
        inputs_perturbed_fwds_agg_batch,
        baselines_perturbed_fwds_agg_batch,
        inputs_fwd_batch,
    )


def aopc(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: Union[
        List[TensorOrTupleOfTensorsGeneric], TensorOrTupleOfTensorsGeneric
    ],
    baselines: BaselineType,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_features_processed_per_batch: Optional[int] = None,
    total_feature_bins: int = 100,
    frozen_features: Optional[List[torch.Tensor]] = None,
    n_random_perms: int = 10,
    seed: Optional[int] = None,
    is_multi_target: bool = False,
    show_progress: bool = False,
    return_intermediate_results: bool = False,
    return_dict: bool = False,
) -> Any:
    """
    Implementation of Area over the Perturbation Curve by Samek et al., 2015. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library.

    Consider a greedy iterative procedure that consists of measuring how the class
    encoded in the image (e.g. as measured by the function f) disappears when we
    progressively remove information from the image x, a process referred to as
    region perturbation, at the specified locations.

    References:
        1) Wojciech Samek et al.: "Evaluating the visualization of what a deep
        neural network has learned." IEEE transactions on neural networks and
        learning systems 28.11 (2016): 2660-2673.

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
                features to be perturbed in each example (`total_feature_bins`) exceeds
                `max_features_processed_per_batch`, they will be sliced
                into batches of `max_features_processed_per_batch` examples and processed
                in a sequential order. However the total effective batch size will still be
                `max_features_processed_per_batch * (2 + n_random_perms)` as in each
                perturbation step, `max_features_processed_per_batch * (2 + n_random_perms)` features
                are processed. If `max_features_processed_per_batch` is None, all
                examples are processed together. `max_features_processed_per_batch` should
                at least be equal `(2 + n_random_perms)` and at most
                `total_feature_bins * (2 + n_random_perms)`.
        total_feature_bins (int, optional): The total number of bins that the features will be perturbed in
                the descending, ascending and random order. Default: 100
        frozen_features (List[torch.Tensor], optional): A list of frozen features that are not perturbed.
                This can be useful for ignoring the input structure features like padding, etc. Default: None
                In case CLS,PAD,SEP tokens are present in the input, they can be frozen by passing the indices
                of feature masks that correspond to these tokens.
        n_random_perms (int, optional): The number of random permutations of the feature importance scores
            that will be used to compute the AOPC scores for the random runs. Default: 10
        seed (int, optional): The seed value for the random number generator for reproducibility. Default: None
        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class.
                Default is False.
        show_progress (bool, optional): Displays the progress of the computation. Default: False
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
                with keys as the metric names and values as the corresponding metric outputs.
                Default is False.
    Returns:
        A dictionary that contains the descending, ascending and random AOPC scores for the input
            samples. The dataclass contains the following fields:
            - desc (Union[List[Tensor], List[List[Tensor]]]): A list of tensors or a list of list of tensors
                representing the descending AOPC scores for each input example. The first dimension is equal to the
                number of examples in the input batch.
            - asc (Union[List[Tensor], List[List[Tensor]]]): A list of tensors or a list of list of tensors
                representing the ascending AOPC scores for each input example. The first dimension is equal to the
                number of examples in the input batch.
            - rand (Union[List[Tensor], List[List[Tensor]]]): A list of tensors or a list of list of tensors
                representing the random AOPC scores for each input example. The first dimension is equal to the
                number of examples in the input batch.

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

        >>> # Computes the aopc scores for saliency maps
        >>> aopc_desc, aopc_asc, aopc_rand = aopc(net, input, attribution, baselines)
    """

    is_attributions_list = isinstance(attributions, list)
    is_targets_list = isinstance(target, list)
    if is_multi_target:
        assert (
            is_attributions_list
        ), "attributions must be a list of tensors or list of tuples of tensors"
        assert is_targets_list, "targets must be a list of targets"
        assert all(isinstance(x, int) for x in target), "targets must be a list of ints"
        assert len(target) == len(attributions), (
            """The number of targets in the targets_list and
            attributions_list must match. Found number of targets in the targets_list is: {} and in the
            attributions_list: {}"""
        ).format(len(target), len(attributions))

    if not is_attributions_list:
        attributions = [attributions]
    if not is_targets_list:
        target = [target]
    inputs_perturbed_aopc_desc_batch = []
    inputs_perturbed_aopc_asc_batch = []
    inputs_perturbed_aopc_rand_batch = []
    baselines_perturbed_aopc_desc_batch = []
    baselines_perturbed_aopc_asc_batch = []
    baselines_perturbed_aopc_rand_batch = []
    inputs_perturbed_fwds_agg_batch = []
    baselines_perturbed_fwds_agg_batch = []
    inputs_fwd_batch = []
    for a, t in zip(attributions, target):
        (
            inputs_perturbed_aopc_desc,
            inputs_perturbed_aopc_asc,
            inputs_perturbed_aopc_rand,
            baselines_perturbed_aopc_desc,
            baselines_perturbed_aopc_asc,
            baselines_perturbed_aopc_rand,
            inputs_perturbed_fwds_agg,
            baselines_perturbed_fwds_agg,
            inputs_fwd,
        ) = _aopc(
            forward_func=forward_func,
            inputs=inputs,
            attributions=a,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            target=t,
            max_features_processed_per_batch=max_features_processed_per_batch,
            total_feature_bins=total_feature_bins,
            frozen_features=frozen_features,
            n_random_perms=n_random_perms,
            seed=seed,
            show_progress=show_progress,
        )
        inputs_perturbed_aopc_desc_batch.append(inputs_perturbed_aopc_desc)
        inputs_perturbed_aopc_asc_batch.append(inputs_perturbed_aopc_asc)
        inputs_perturbed_aopc_rand_batch.append(inputs_perturbed_aopc_rand)
        baselines_perturbed_aopc_desc_batch.append(baselines_perturbed_aopc_desc)
        baselines_perturbed_aopc_asc_batch.append(baselines_perturbed_aopc_asc)
        baselines_perturbed_aopc_rand_batch.append(baselines_perturbed_aopc_rand)
        inputs_perturbed_fwds_agg_batch.append(inputs_perturbed_fwds_agg)
        baselines_perturbed_fwds_agg_batch.append(baselines_perturbed_fwds_agg)
        inputs_fwd_batch.append(inputs_fwd)

    if not is_attributions_list:
        inputs_perturbed_aopc_desc_batch = inputs_perturbed_aopc_desc_batch[0]
        inputs_perturbed_aopc_asc_batch = inputs_perturbed_aopc_asc_batch[0]
        inputs_perturbed_aopc_rand_batch = inputs_perturbed_aopc_rand_batch[0]
        baselines_perturbed_aopc_desc_batch = baselines_perturbed_aopc_desc_batch[0]
        baselines_perturbed_aopc_asc_batch = baselines_perturbed_aopc_asc_batch[0]
        baselines_perturbed_aopc_rand_batch = baselines_perturbed_aopc_rand_batch[0]
        inputs_perturbed_fwds_agg_batch = inputs_perturbed_fwds_agg_batch[0]
        baselines_perturbed_fwds_agg_batch = baselines_perturbed_fwds_agg_batch[0]
        inputs_fwd_batch = inputs_fwd_batch[0]

    if return_intermediate_results:
        if return_dict:
            return dict(
                desc=inputs_perturbed_aopc_desc_batch,
                asc=inputs_perturbed_aopc_asc_batch,
                rand=inputs_perturbed_aopc_rand_batch,
                baselines_perturbed_desc=baselines_perturbed_aopc_desc_batch,
                baselines_perturbed_asc=baselines_perturbed_aopc_asc_batch,
                baselines_perturbed_rand=baselines_perturbed_aopc_rand_batch,
                inputs_perturbed_fwds_agg_batch=inputs_perturbed_fwds_agg_batch,
                baselines_perturbed_fwds_agg_batch=baselines_perturbed_fwds_agg_batch,
                inputs_fwd_batch=inputs_fwd_batch,
            )
        else:
            return (
                inputs_perturbed_aopc_desc_batch,
                inputs_perturbed_aopc_asc_batch,
                inputs_perturbed_aopc_rand_batch,
                baselines_perturbed_aopc_desc_batch,
                baselines_perturbed_aopc_asc_batch,
                baselines_perturbed_aopc_rand_batch,
                inputs_perturbed_fwds_agg_batch,
                baselines_perturbed_fwds_agg_batch,
                inputs_fwd_batch,
            )
    else:
        if return_dict:
            return dict(
                desc=inputs_perturbed_aopc_desc_batch,
                asc=inputs_perturbed_aopc_asc_batch,
                rand=inputs_perturbed_aopc_rand_batch,
                baselines_perturbed_desc=baselines_perturbed_aopc_desc_batch,
                baselines_perturbed_asc=baselines_perturbed_aopc_asc_batch,
                baselines_perturbed_rand=baselines_perturbed_aopc_rand_batch,
            )
        else:
            return (
                inputs_perturbed_aopc_desc_batch,
                inputs_perturbed_aopc_asc_batch,
                inputs_perturbed_aopc_rand_batch,
                baselines_perturbed_aopc_desc_batch,
                baselines_perturbed_aopc_asc_batch,
                baselines_perturbed_aopc_rand_batch,
            )
