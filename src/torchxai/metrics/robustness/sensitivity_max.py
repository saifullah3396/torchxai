from copy import deepcopy
from inspect import signature
from typing import Any, Callable, Tuple, Union, cast

import torch
from captum._utils.common import (
    _expand_and_update_additional_forward_args,
    _expand_and_update_baselines,
    _expand_and_update_feature_mask,
    _expand_and_update_target,
    _format_baseline,
    _format_tensor_into_tuples,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.metrics._utils.batching import _divide_and_aggregate_metrics
from torch import Tensor


def default_perturb_func(
    inputs: TensorOrTupleOfTensorsGeneric, perturb_radius: float = 0.02
) -> Tuple[Tensor, ...]:
    r"""A default function for generating perturbations of `inputs`
    within perturbation radius of `perturb_radius`.
    This function samples uniformly random from the L_Infinity ball
    with `perturb_radius` radius.
    The users can override this function if they prefer to use a
    different perturbation function.

    Args:

        inputs (Tensor or tuple[Tensor, ...]): The input tensors that we'd
                like to perturb by adding a random noise sampled uniformly
                random from an L_infinity ball with a radius `perturb_radius`.

        radius (float): A radius used for sampling from
                an L_infinity ball.

    Returns:

        perturbed_input (tuple[Tensor, ...]): A list of perturbed inputs that
                are created by adding noise sampled uniformly random
                from L_infiniy ball with a radius `perturb_radius` to the
                original inputs.

    """
    inputs = _format_tensor_into_tuples(inputs)
    perturbed_input = tuple(
        input
        + torch.FloatTensor(input.size())  # type: ignore
        .uniform_(-perturb_radius, perturb_radius)
        .to(input.device)
        for input in inputs
    )
    return perturbed_input


def sensitivity_max(
    explanation_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    perturb_func: Callable = default_perturb_func,
    perturb_radius: float = 0.02,
    n_perturb_samples: int = 10,
    norm_ord: str = "fro",
    max_examples_per_batch: int = None,
    **kwargs: Any,
) -> Tensor:
    """
    This is a modified version of the captum `sensitivity_max` (see: from captum.metrics import sensitivity_max)
    function that repeats the feature masks when performing perturbations.
    """

    def _generate_perturbations(
        current_n_perturb_samples: int,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        The perturbations are generated for each example
        `current_n_perturb_samples` times.

        For perfomance reasons we are not calling `perturb_func` on each example but
        on a batch that contains `current_n_perturb_samples` repeated instances
        per example.
        """
        inputs_expanded: Union[Tensor, Tuple[Tensor, ...]] = tuple(
            torch.repeat_interleave(input, current_n_perturb_samples, dim=0)
            for input in inputs
        )
        if len(inputs_expanded) == 1:
            inputs_expanded = inputs_expanded[0]

        return (
            perturb_func(inputs_expanded, perturb_radius)
            if len(signature(perturb_func).parameters) > 1
            else perturb_func(inputs_expanded)
        )

    def max_values(input_tnsr: Tensor) -> Tensor:
        return torch.max(input_tnsr, dim=1).values  # type: ignore

    kwarg_expanded_for = None
    kwargs_copy: Any = None

    def _next_sensitivity_max(current_n_perturb_samples: int) -> Tensor:
        inputs_perturbed = _generate_perturbations(current_n_perturb_samples)

        # copy kwargs and update some of the arguments that need to be expanded
        nonlocal kwarg_expanded_for
        nonlocal kwargs_copy
        if (
            kwarg_expanded_for is None
            or kwarg_expanded_for != current_n_perturb_samples
        ):
            kwarg_expanded_for = current_n_perturb_samples
            kwargs_copy = deepcopy(kwargs)
            _expand_and_update_additional_forward_args(
                current_n_perturb_samples, kwargs_copy
            )
            _expand_and_update_feature_mask(current_n_perturb_samples, kwargs_copy)
            _expand_and_update_target(current_n_perturb_samples, kwargs_copy)
            if "baselines" in kwargs:
                baselines = kwargs["baselines"]
                baselines = _format_baseline(
                    baselines, cast(Tuple[Tensor, ...], inputs)
                )
                if (
                    isinstance(baselines[0], Tensor)
                    and baselines[0].shape == inputs[0].shape
                ):
                    _expand_and_update_baselines(
                        cast(Tuple[Tensor, ...], inputs),
                        current_n_perturb_samples,
                        kwargs_copy,
                    )

        expl_perturbed_inputs = explanation_func(inputs_perturbed, **kwargs_copy)

        # tuplize `expl_perturbed_inputs` in case it is not
        expl_perturbed_inputs = _format_tensor_into_tuples(expl_perturbed_inputs)

        expl_inputs_expanded = tuple(
            expl_input.repeat_interleave(current_n_perturb_samples, dim=0)
            for expl_input in expl_inputs
        )

        sensitivities = torch.cat(
            [
                (expl_input - expl_perturbed).view(expl_perturbed.size(0), -1)
                for expl_perturbed, expl_input in zip(
                    expl_perturbed_inputs, expl_inputs_expanded
                )
            ],
            dim=1,
        )
        # compute the norm of original input explanations
        expl_inputs_norm_expanded = torch.norm(
            torch.cat(
                [expl_input.view(expl_input.size(0), -1) for expl_input in expl_inputs],
                dim=1,
            ),
            p=norm_ord,
            dim=1,
            keepdim=True,
        ).repeat_interleave(current_n_perturb_samples, dim=0)
        expl_inputs_norm_expanded = torch.where(
            expl_inputs_norm_expanded == 0.0,
            torch.tensor(
                1.0,
                device=expl_inputs_norm_expanded.device,
                dtype=expl_inputs_norm_expanded.dtype,
            ),
            expl_inputs_norm_expanded,
        )

        # compute the norm for each input noisy example
        sensitivities_norm = (
            torch.norm(sensitivities, p=norm_ord, dim=1, keepdim=True)
            / expl_inputs_norm_expanded
        )
        return max_values(sensitivities_norm.view(bsz, -1))

    inputs = _format_tensor_into_tuples(inputs)  # type: ignore

    bsz = inputs[0].size(0)

    with torch.no_grad():
        expl_inputs = explanation_func(inputs, **kwargs)
        metrics_max = _divide_and_aggregate_metrics(
            cast(Tuple[Tensor, ...], inputs),
            n_perturb_samples,
            _next_sensitivity_max,
            max_examples_per_batch=max_examples_per_batch,
            agg_func=torch.max,
        )
    return metrics_max
