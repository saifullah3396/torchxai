#!/usr/bin/env python3

from typing import Any, Callable, Tuple, Union, cast

import numpy as np
import torch
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
from captum.log import log_usage
from torch import Tensor

from torchxai.metrics._utils.batching import (
    _divide_and_aggregate_metrics_n_features,
    _divide_and_aggregate_metrics_n_perturbations_per_feature,
)
from torchxai.metrics._utils.common import (
    _construct_default_feature_masks,
    _draw_perturbated_inputs,
    _draw_perturbated_inputs_with_splits,
    _format_tensor_feature_dim,
    _reduce_tensor_with_indices,
    _split_tensors_to_tuple_tensors,
    _tuple_tensors_to_tensors,
    _validate_feature_mask,
)


def perturb_input(input, baseline, feature_mask, indices, feature_idx):
    # for each feature in the current step incrementally replace the baseline with the original sample
    perturbation_mask = feature_mask == indices[feature_idx]
    input[perturbation_mask] = baseline[
        perturbation_mask
    ]  # input[0] here since batch size is 1


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


@log_usage()
def eval_aopcs_single_sample(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    feature_masks: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_examples_per_batch: int = None,
    total_features_perturbed: int = 100,
    n_random_perms: int = 10,
    seed: int = 0,
) -> Tensor:
    def _next_aopc_tensors(
        current_n_perturbed_features: int,
        current_n_steps: int,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]:
        # get the indices of features that will be perturbed in the current iteration
        # for example if we do 1 by 1 perturbation, then the first iteration will perturb the first feature
        # the second iteration will perturb both the first and second feature and so on

        perturbed_inputs = []
        for feature_idx in range(
            current_n_steps - current_n_perturbed_features, current_n_steps
        ):
            # make a global perturbed input tensor for each perturbation type, ascending, descending and random
            for input, indices in zip(
                [
                    global_perturbed_inputs_desc,  # this order is maintained in outputs
                    global_perturbed_inputs_asc,  # this order is maintained in outputs
                    *global_perturbed_inputs_rand,  # this order is maintained in outputs
                ],
                [
                    descending_attribution_indices,  # this order is maintained in outputs
                    ascending_attribution_indices,  # this order is maintained in outputs
                    *rand_attribution_indices,  # this order is maintained in outputs
                ],
            ):
                perturb_input(
                    input=input,
                    baseline=baselines,
                    feature_mask=feature_masks,
                    indices=indices,
                    feature_idx=feature_idx,
                )
                perturbed_inputs.append(input.clone())
        perturbed_inputs = torch.cat(perturbed_inputs)

        # this is for debugging purposes, to see the perturbed inputs in a matrix form
        # for every 12 inputs in the batch, first is descending, second is ascending and the rest are random
        # and then this repeats for the next 12 inputs
        # for example a minimum batch size will always be 12 for current_n_perturbed_features=1
        # print some info for indices,
        # print(
        #     "descending",
        #     descending_attribution_indices[
        #         current_n_steps - current_n_perturbed_features : current_n_steps
        #     ],
        # )
        # print(
        #     "ascending",
        #     ascending_attribution_indices[
        #         current_n_steps - current_n_perturbed_features : current_n_steps
        #     ],
        # )
        # print(
        #     "rand",
        #     rand_attribution_indices[
        #         :,
        #         current_n_steps - current_n_perturbed_features : current_n_steps,
        #     ],
        # )
        # _draw_perturbated_inputs(perturbed_inputs=perturbed_inputs)
        # _draw_perturbated_inputs_with_splits(
        #     perturbed_inputs=perturbed_inputs, inputs_shape=inputs_shape
        # )

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

        # split the perturbed inputs by feature size
        perturbed_inputs = _split_tensors_to_tuple_tensors(
            perturbed_inputs, inputs_shape, dim=1
        )
        inputs_perturbed_fwd = _run_forward(
            forward_func,
            # the inputs are [batch_size, feature_size, feature_dims] so we need to split them by feature size
            perturbed_inputs,
            targets_expanded,
            additional_forward_args_expanded,
        )

        # we expect a single output per batch
        inputs_perturbed_fwd = inputs_perturbed_fwd.squeeze(-1)

        # reshape outputs to [desc, asc, rand * n_random_perms]
        inputs_perturbed_fwd = torch.stack(
            inputs_perturbed_fwd.split(2 + n_random_perms), dim=0
        )

        # this is a list of tensors which holds the forward outputs of the model
        # on each feature group perturbation
        # the first element will be when the feature with lowest importance is added to the baseline
        # the last element will be when all features are added to the baseline
        return inputs_perturbed_fwd

    def _cat_aopc_tensors(agg_tensors, tensors):
        return torch.cat([agg_tensors, tensors], dim=0)

    with torch.no_grad():
        bsz = inputs[0].size(0)
        assert bsz == 1, "Batch size must be 1 for aopc_single_sample"

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

        # get the gathererd-attributions sorted in ascending order of their importance
        ascending_attribution_indices = torch.argsort(reduced_attributions)[
            :total_features_perturbed
        ]

        # get the gathererd-attributions sorted in descending order of their importance
        descending_attribution_indices = torch.argsort(
            reduced_attributions, descending=True
        )[:total_features_perturbed]

        # get the gathererd-attributions sorted in a random order of their importance n times
        generator = torch.Generator().manual_seed(seed)
        rand_attribution_indices = torch.stack(
            [
                torch.randperm(
                    len(reduced_attributions),
                    generator=generator,
                )[:total_features_perturbed]
                for _ in range(n_random_perms)
            ]
        )

        assert ascending_attribution_indices.max() < n_features
        assert descending_attribution_indices.max() < n_features
        assert rand_attribution_indices.max() < n_features

        # make a global perturbed input tensor for each perturbation type, ascending, descending and random
        global_perturbed_inputs_desc = inputs.clone()
        global_perturbed_inputs_asc = inputs.clone()
        global_perturbed_inputs_rand = [inputs.clone() for _ in range(n_random_perms)]

        # the logic for this implementation as as follows:
        # we start from baseline and in each iteration, a feature group is replaced by the original sample
        # in ascending order of its importance
        inputs_perturbed_fwds_agg = (
            _divide_and_aggregate_metrics_n_perturbations_per_feature(
                n_perturbations_per_feature=(
                    2 + n_random_perms
                ),  # 2 for ascending and descending and n_random_perms for random
                n_features=min(total_features_perturbed, n_features),
                metric_func=_next_aopc_tensors,
                agg_func=_cat_aopc_tensors,
                max_examples_per_batch=max_examples_per_batch,
            )
        )
        aopc_scores = compute_aopc_scores_vectorized(
            inputs_perturbed_fwds_agg.T, inputs_fwd
        )

    return aopc_scores


@log_usage()
def aopc(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType,
    feature_masks: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    max_examples_per_batch: int = None,
    total_features_perturbed: int = 100,
    n_random_perms: int = 10,
    seed: int = 0,
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
    aopc_batch = []
    for sample_idx in range(bsz):
        aopc_scores = eval_aopcs_single_sample(
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
            total_features_perturbed=total_features_perturbed,
            n_random_perms=n_random_perms,
            seed=seed,
        )
        aopc_batch.append(aopc_scores)
    aopc_batch = torch.stack(aopc_batch)
    return (
        aopc_batch[:, 0, :],
        aopc_batch[:, 1, :],
        aopc_batch[:, 2:, :].mean(1),
    )  # descending, ascending, random


# This is for testing code
# import torch
# from torch import nn

# torch.manual_seed(0)


# class DummyModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, a, b, c, d):
#         bs = a.shape[0]
#         return a[:, :, 0]


# model = DummyModel()

# import numpy as np
# import torch

# dummy_input = (
#     torch.randn(2, 12, 768),
#     torch.randn(2, 12, 768),
#     torch.randn(2, 12, 768),
#     torch.randn(2, 18, 768),
# )
# dummy_baseline = (
#     torch.zeros(2, 12, 768),
#     torch.zeros(2, 12, 768),
#     torch.zeros(2, 12, 768),
#     torch.zeros(2, 18, 768),
# )
# dummy_attr = tuple(x.sum(-1) for x in dummy_input)


# def example_feature_mask(start_idx, dim=12):
#     feature_group_size = 1
#     return (
#         torch.arange(0, dim, feature_group_size).repeat_interleave(feature_group_size)[
#             :dim
#         ]
#         / feature_group_size
#     ).long().unsqueeze(0) + start_idx


# token_feature_masks = example_feature_mask(start_idx=0)
# position_feature_masks = example_feature_mask(start_idx=token_feature_masks.max() + 1)
# bbox_feature_masks = example_feature_mask(start_idx=position_feature_masks.max() + 1)
# image_feature_masks = example_feature_mask(
#     start_idx=bbox_feature_masks.max() + 1, dim=18
# )
# # image_feature_masks =  torch.arange(196).unsqueeze(0)+bbox_feature_masks.max()+1

# dummy_feature_masks = (
#     token_feature_masks.repeat(2, 1),
#     position_feature_masks.repeat(2, 1),
#     bbox_feature_masks.repeat(2, 1),
#     image_feature_masks.repeat(2, 1),
# )
# dummy_feature_masks_shapes = tuple(x.shape for x in dummy_feature_masks)

# aopc(
#     forward_func=model,
#     inputs=dummy_input,
#     attributions=dummy_attr,
#     baselines=dummy_baseline,
#     feature_masks=dummy_feature_masks,
#     target=torch.tensor([0, 0]),
#     max_examples_per_batch=24,
# )
