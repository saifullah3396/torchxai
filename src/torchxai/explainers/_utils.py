#!/usr/bin/env python3
import warnings
from inspect import signature
from typing import Any, Callable, Tuple, Union

import torch
from captum._utils.common import (
    ExpansionTypes,
    _expand_target,
    _format_additional_forward_args,
    _format_inputs,
    _select_targets,
)
from captum._utils.typing import TargetType
from captum.attr._utils.approximation_methods import approximation_parameters
from torch import Tensor


def _generate_mask_weights(feature_mask_batch: torch.Tensor) -> torch.Tensor:
    """
    This function takes a batch of feature masks and generates a corresponding
    batch of weighted feature masks. Each unique feature in the mask is assigned
    a weight that is inversely proportional to its frequency in the mask.
    Args:
        feature_mask_batch (torch.Tensor): A batch of feature masks with shape (batch_size, ...), where each element
            is an integer representing a feature.
    Returns:
        torch.Tensor: A batch of weighted feature masks with the same shape as `feature_mask_batch`, where each
            feature is weighted by the inverse of its frequency in the mask.
    """

    feature_mask_weighted_batch = torch.zeros_like(
        feature_mask_batch, dtype=torch.float
    )
    for feature_mask, feature_mask_weighted in zip(
        feature_mask_batch, feature_mask_weighted_batch
    ):  # batch iteration
        labels, counts = torch.unique(feature_mask, return_counts=True)
        for idx in range(labels.shape[0]):
            feature_mask_weighted[feature_mask == labels[idx]] = 1.0 / counts[idx]
    return feature_mask_weighted_batch


def _run_forward_multi_target(
    forward_func: Callable,
    inputs: Any,
    target: Tuple[TargetType, ...] = None,
    additional_forward_args: Any = None,
) -> Tensor:
    forward_func_args = signature(forward_func).parameters
    if len(forward_func_args) == 0:
        output = forward_func()
        return output if target is None else _select_targets(output, target)

    # make everything a tuple so that it is easy to unpack without
    # using if-statements
    inputs = _format_inputs(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)

    output = forward_func(
        *(
            (*inputs, *additional_forward_args)
            if additional_forward_args is not None
            else inputs
        )
    )
    if isinstance(target, (tuple, list)):
        return torch.stack(
            [_select_targets(output, single_target) for single_target in target], dim=1
        )
    else:
        return _select_targets(output, target)


def _compute_gradients_sequential_autograd(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target: Tuple[TargetType, ...] = None,
    additional_forward_args: Any = None,
) -> Tuple[Tensor, ...]:
    with torch.autograd.set_grad_enabled(True):
        outputs = _run_forward_multi_target(
            forward_fn, inputs, target, additional_forward_args
        )
        if (
            len(outputs.shape) > 1
        ):  # this is the case where the output is of dimension [batch_size, ...]
            unit_vectors = torch.eye(outputs.shape[-1]).to(outputs.device)
            multi_target_gradients = []
            for idx, v in enumerate(unit_vectors):
                retain_graph = idx < len(unit_vectors) - 1

                # autograd returns a tuple of gradients, for a tuple of inputs against each output
                grads = torch.autograd.grad(
                    torch.unbind(outputs),
                    inputs,
                    (v,) * outputs.shape[0],
                    retain_graph=retain_graph,
                )
                multi_target_gradients.append(grads)
        else:
            # this is the case where the output is of dimension [batch_size]
            # in which case there is not target to choose from
            grads = torch.autograd.grad(
                torch.unbind(outputs),
                inputs,
            )
            multi_target_gradients = [grads]

    return multi_target_gradients


def _compute_gradients_vmap_autograd(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target: Tuple[TargetType, ...] = None,
    additional_forward_args: Any = None,
) -> Tuple[Tensor, ...]:
    with torch.autograd.set_grad_enabled(True):
        outputs = _run_forward_multi_target(
            forward_fn, inputs, target, additional_forward_args
        )

        if (
            len(outputs.shape) > 1
        ):  # this is the case where the output is of dimension [batch_size, ...]
            unit_vectors = torch.eye(outputs.shape[-1]).to(outputs.device)

            def get_vjp(v):
                return torch.autograd.grad(
                    torch.unbind(outputs),
                    inputs,
                    (v,) * outputs.shape[0],
                    retain_graph=True,
                )

            multi_target_gradients = torch.vmap(get_vjp, chunk_size=1)(unit_vectors)
            multi_target_gradients = torch.stack(multi_target_gradients).swapaxes(0, 1)
            return [
                tuple(grad_per_input for grad_per_input in grad_per_target)
                for grad_per_target in multi_target_gradients
            ]
        else:
            # this is the case where the output is of dimension [batch_size]
            # in which case there is not target to choose from
            grads = torch.autograd.grad(
                torch.unbind(outputs),
                inputs,
            )
            multi_target_gradients = [grads]
    return [
        tuple(grad_per_input for grad_per_input in grad_per_target)
        for grad_per_target in multi_target_gradients
    ]


def _batch_attribution_multi_target(
    attr_method,
    num_examples,
    internal_batch_size,
    n_steps,
    include_endpoint=False,
    **kwargs,
):
    """
    This method applies internal batching to given attribution method, dividing
    the total steps into batches and running each independently and sequentially,
    adding each result to compute the total attribution.

    Step sizes and alphas are spliced for each batch and passed explicitly for each
    call to _attribute.

    kwargs include all argument necessary to pass to each attribute call, except
    for n_steps, which is computed based on the number of steps for the batch.

    include_endpoint ensures that one step overlaps between each batch, which
    is necessary for some methods, particularly LayerConductance.
    """
    if internal_batch_size < num_examples:
        warnings.warn(
            "Internal batch size cannot be less than the number of input examples. "
            "Defaulting to internal batch size of %d equal to the number of examples."
            % num_examples
        )
    # Number of steps for each batch
    step_count = max(1, internal_batch_size // num_examples)
    if include_endpoint:
        if step_count < 2:
            step_count = 2
            warnings.warn(
                "This method computes finite differences between evaluations at "
                "consecutive steps, so internal batch size must be at least twice "
                "the number of examples. Defaulting to internal batch size of %d"
                " equal to twice the number of examples." % (2 * num_examples)
            )

    total_attr = None
    cumulative_steps = 0
    step_sizes_func, alphas_func = approximation_parameters(kwargs["method"])
    full_step_sizes = step_sizes_func(n_steps)
    full_alphas = alphas_func(n_steps)

    while cumulative_steps < n_steps:
        start_step = cumulative_steps
        end_step = min(start_step + step_count, n_steps)
        batch_steps = end_step - start_step

        if include_endpoint:
            batch_steps -= 1

        step_sizes = full_step_sizes[start_step:end_step]
        alphas = full_alphas[start_step:end_step]
        current_attr = attr_method._attribute(
            **kwargs, n_steps=batch_steps, step_sizes_and_alphas=(step_sizes, alphas)
        )

        if total_attr is None:
            total_attr = current_attr
        else:
            for output_idx in range(len(total_attr)):
                if isinstance(total_attr[output_idx], Tensor):
                    total_attr[output_idx] = (
                        total_attr[output_idx] + current_attr.detach()
                    )
                else:
                    total_attr[output_idx] = tuple(
                        current.detach() + prev_total
                        for current, prev_total in zip(
                            current_attr[output_idx], total_attr[output_idx]
                        )
                    )

        if include_endpoint and end_step < n_steps:
            cumulative_steps = end_step - 1
        else:
            cumulative_steps = end_step
    return total_attr


def _verify_target_for_multi_target_impl(inputs, target):
    bsz = inputs[0].shape[0]
    if target is not None and isinstance(target, (list, tuple)):
        for t in target:
            if isinstance(t, (list, tuple)):
                assert (
                    len(t) == bsz
                ), "Each target in the list must be a tensor of size batch size"


def _expand_and_update_target_multi_target(n_samples: int, kwargs: dict):
    if "target" not in kwargs:
        return
    target = kwargs["target"]
    if isinstance(target, list):
        target = [
            _expand_target(
                t, n_samples, expansion_type=ExpansionTypes.repeat_interleave
            )
            for t in target
        ]
    else:
        target = _expand_target(
            target, n_samples, expansion_type=ExpansionTypes.repeat_interleave
        )

    # update kwargs with expanded baseline
    kwargs["target"] = target


def _expand_feature_mask_to_target(
    feature_mask: Tuple[torch.Tensor], inputs: Tuple[torch.Tensor]
) -> Tuple[torch.Tensor]:
    """
    Expands each feature mask tensor to match the shape of the corresponding input tensor.
    Args:
        feature_mask (Tuple[torch.Tensor]): A tuple of tensors representing the feature masks.
        inputs (Tuple[torch.Tensor]): A tuple of tensors representing the inputs.
    Returns:
        Tuple[torch.Tensor]: A tuple of tensors where each feature mask is expanded to match the shape of the
            corresponding input tensor.
    """
    if feature_mask is None:
        return feature_mask

    return_first_element = False
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    if not isinstance(feature_mask, tuple):
        feature_mask = (feature_mask,)
        return_first_element = False

    feature_mask = tuple(
        (
            mask.unsqueeze(-1).expand_as(input)
            if len(mask.shape) < len(input.shape)
            else mask.expand_as(input)
        )
        for input, mask in zip(inputs, feature_mask)
    )
    if return_first_element:
        return feature_mask[0]
    return feature_mask
