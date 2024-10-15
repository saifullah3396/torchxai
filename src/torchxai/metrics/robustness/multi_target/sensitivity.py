from copy import deepcopy
from inspect import signature
from typing import Any, Callable, List, Tuple, Union, cast

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

from torchxai.explainers.explainer import Explainer
from torchxai.metrics.robustness.utilities import default_perturb_func


def _multi_target_sensitivity_scores(
    explanation_func: Explainer,
    inputs: TensorOrTupleOfTensorsGeneric,
    perturb_func: Callable = default_perturb_func,
    perturb_radius: float = 0.02,
    n_perturb_samples: int = 10,
    norm_ord: str = "fro",
    max_examples_per_batch: int = None,
    **kwargs: Any,
) -> List[Tensor]:

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

        def compute_sensitivity_per_target(expl_inputs, expl_perturbed_inputs):
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
                    [
                        expl_input.view(expl_input.size(0), -1)
                        for expl_input in expl_inputs
                    ],
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

        # this computes the explanation for the original input for all targets in a single call
        expl_perturbed_inputs_list = explanation_func(inputs_perturbed, **kwargs_copy)

        return [
            compute_sensitivity_per_target(expl_inputs, expl_perturbed)
            for expl_inputs, expl_perturbed in zip(
                expl_inputs_list, expl_perturbed_inputs_list
            )
        ]

    assert isinstance(explanation_func, Explainer), (
        "Explanation function must be an instance of "
        "`torchxai.explainers.Explainer`."
    )
    assert (
        explanation_func._is_multi_target
    ), "Explanation function must be a multi-target explainer."
    target = kwargs.get("target", None)
    assert isinstance(target, list), "targets must be a list of targets"
    assert all(isinstance(x, int) for x in target), "targets must be a list of ints"

    inputs = _format_tensor_into_tuples(inputs)  # type: ignore

    bsz = inputs[0].size(0)

    def _agg_sensitivity_scores(agg_tensors, tensors):
        return [torch.cat([agg_t, t], dim=-1) for agg_t, t in zip(agg_tensors, tensors)]

    with torch.no_grad():
        expl_inputs_list = explanation_func.explain(inputs, **kwargs)
        metric_scores_list = _divide_and_aggregate_metrics(
            cast(Tuple[Tensor, ...], inputs),
            n_perturb_samples,
            _next_sensitivity_max,
            max_examples_per_batch=max_examples_per_batch,
            agg_func=_agg_sensitivity_scores,
        )
    return metric_scores_list
