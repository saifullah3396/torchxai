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
from torchxai.metrics.robustness.multi_target.sensitivity import (
    _multi_target_sensitivity_scores,
)
from torchxai.metrics.robustness.utilities import default_perturb_func
from captum.attr import Attribution


def _sensitivity_scores(
    explanation_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    perturb_func: Callable = default_perturb_func,
    perturb_radius: float = 0.02,
    n_perturb_samples: int = 10,
    norm_ord: str = "fro",
    max_examples_per_batch: int = None,
    is_multi_target: bool = False,
    **kwargs: Any,
) -> Union[Tensor, List[Tensor]]:
    if is_multi_target:
        return _multi_target_sensitivity_scores(
            explanation_func,
            inputs,
            perturb_func=perturb_func,
            perturb_radius=perturb_radius,
            n_perturb_samples=n_perturb_samples,
            norm_ord=norm_ord,
            max_examples_per_batch=max_examples_per_batch,
            **kwargs,
        )

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
        return sensitivities_norm.view(bsz, -1)

    def _agg_tensors(agg_tensors, tensors):
        return torch.cat((agg_tensors, tensors), dim=1)

    inputs = _format_tensor_into_tuples(inputs)  # type: ignore

    bsz = inputs[0].size(0)

    with torch.no_grad():
        expl_inputs = explanation_func(inputs, **kwargs)
        scores = _divide_and_aggregate_metrics(
            cast(Tuple[Tensor, ...], inputs),
            n_perturb_samples,
            _next_sensitivity_max,
            max_examples_per_batch=max_examples_per_batch,
            agg_func=_agg_tensors,
        )
    return scores


def sensitivity_max_and_avg(
    explainer: Union[Explainer, Attribution],
    inputs: TensorOrTupleOfTensorsGeneric,
    perturb_func: Callable = default_perturb_func,
    perturb_radius: float = 0.02,
    n_perturb_samples: int = 10,
    norm_ord: str = "fro",
    max_examples_per_batch: int = None,
    is_multi_target: bool = False,
    return_dict: bool = False,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""
    This is a modified version of the captum `captum.metric.sensitivity_max` (see: from captum.metrics import sensitivity_max)
    function that repeats the feature masks when performing perturbations. This function returns the sensitivity
    scores with both average and max aggregation resulting in sensitivity_max and sensitivity_avg together.

    The metric returns a list of senstivity scores if is_multi_target is True using the
    `torchxai.metrics.robustness.multi_target._multi_target_sensitivity_scores`,
    otherwise it returns a single sensitivity score same as the `captum.metric.sensitivity_max`.

    Explanation sensitivity measures the extent of explanation change when
    the input is slightly perturbed. It has been shown that the models that
    have high explanation sensitivity are prone to adversarial attacks:
    `Interpretation of Neural Networks is Fragile`
    https://www.aaai.org/ojs/index.php/AAAI/article/view/4252

    `sensitivity_max` metric measures maximum sensitivity of an explanation
    using Monte Carlo sampling-based approximation. By default in order to
    do so it samples multiple data points from a sub-space of an L-Infinity
    ball that has a `perturb_radius` radius using `default_perturb_func`
    default perturbation function. In a general case users can
    use any L_p ball or any other custom sampling technique that they
    prefer by providing a custom `perturb_func`.

    Note that max sensitivity is similar to Lipschitz Continuity metric
    however it is more robust and easier to estimate.
    Since the explanation, for instance an attribution function,
    may not always be continuous, can lead to unbounded
    Lipschitz continuity. Therefore the latter isn't always appropriate.

    More about the Lipschitz Continuity Metric can also be found here
    `On the Robustness of Interpretability Methods`
    https://arxiv.org/abs/1806.08049
    and
    `Towards Robust Interpretability with Self-Explaining Neural Networks`
    https://papers.nips.cc/paper\
    8003-towards-robust-interpretability-
    with-self-explaining-neural-networks.pdf

    More details about sensitivity max can be found here:
    `On the (In)fidelity and Sensitivity of Explanations`
    https://arxiv.org/abs/1901.09392

    Args:

        explanation_func (Callable):
                This function can be the `attribute` method of an
                attribution algorithm or any other explanation method
                that returns the explanations.

        inputs (Tensor or tuple[Tensor, ...]): Input for which
                explanations are computed. If `explanation_func` takes a
                single tensor as input, a single input tensor should
                be provided.
                If `explanation_func` takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples (aka batch size), and if
                multiple input tensors are provided, the examples must
                be aligned appropriately.

        perturb_func (Callable):
                The perturbation function of model inputs. This function takes
                model inputs and optionally `perturb_radius` if
                the function takes more than one argument and returns
                perturbed inputs.

                If there are more than one inputs passed to sensitivity function those
                will be passed to `perturb_func` as tuples in the same order as they
                are passed to sensitivity function.

                It is important to note that for performance reasons `perturb_func`
                isn't called for each example individually but on a batch of
                input examples that are repeated `max_examples_per_batch / batch_size`
                times within the batch.

            Default: default_perturb_func
        perturb_radius (float, optional): The epsilon radius used for sampling.
            In the `default_perturb_func` it is used as the radius of
            the L-Infinity ball. In a general case it can serve as a radius of
            any L_p norm.
            This argument is passed to `perturb_func` if it takes more than
            one argument.

            Default: 0.02
        n_perturb_samples (int, optional): The number of times input tensors
                are perturbed. Each input example in the inputs tensor is
                expanded `n_perturb_samples` times before calling
                `perturb_func` function.

                Default: 10
        norm_ord (int, float, or str, optional): The type of norm that is used to
                compute the norm of the sensitivity matrix which is defined as the
                difference between the explanation function at its input and perturbed
                input. Acceptable values are either a string of 'fro' or 'nuc', or a
                number in the range of [-inf, inf] (including float("-inf") &
                float("inf")).

                Default: 'fro'
        max_examples_per_batch (int, optional): The number of maximum input
                examples that are processed together. In case the number of
                examples (`input batch size * n_perturb_samples`) exceeds
                `max_examples_per_batch`, they will be sliced
                into batches of `max_examples_per_batch` examples and processed
                in a sequential order. If `max_examples_per_batch` is None, all
                examples are processed together. `max_examples_per_batch` should
                at least be equal `input batch size` and at most
                `input batch size * n_perturb_samples`.

                Default: None
        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class. For multi-target
                infidelity, captum implementation is extened in _multi_target_infidelity function.
                Default is False.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
                with keys as the metric names and values as the corresponding metric outputs.
                Default is False.
        **kwargs (Any, optional): Contains a list of arguments that are passed
                to `explanation_func` explanation function which in some cases
                could be the `attribute` function of an attribution algorithm.
                Any additional arguments that need be passed to the explanation
                function should be included here.
                For instance, such arguments include:
                `additional_forward_args`, `baselines` and `target`.

    Returns:

        sensitivities (Tensor): A tensor of scalar sensitivity scores per
                input example. The first dimension is equal to the
                number of examples in the input batch and the second
                dimension is one. Returned sensitivities are normalized by
                the magnitudes of the input explanations.

    Examples::
        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> saliency = Saliency(net)
        >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> # Computes sensitivity score for saliency maps of class 3
        >>> sens = sensitivity_max(saliency.attribute, input, target = 3)
    """
    assert isinstance(explainer, (Explainer, Attribution)), (
        "The explainer must be an instance of the Explainer class or "
        "the Atribution class from the captum library."
    )
    explanation_func = (
        explainer.explain if isinstance(explainer, Explainer) else explainer.attribute
    )
    sensitivity_scores = _sensitivity_scores(
        explanation_func,
        inputs,
        perturb_func=perturb_func,
        perturb_radius=perturb_radius,
        n_perturb_samples=n_perturb_samples,
        norm_ord=norm_ord,
        max_examples_per_batch=max_examples_per_batch,
        is_multi_target=is_multi_target,
        **kwargs,
    )

    if is_multi_target:
        if return_dict:
            return {
                "sensitivity_max": [
                    torch.max(score, dim=1) for score in sensitivity_scores
                ],
                "sensitivity_avg": [
                    torch.mean(score, dim=1) for score in sensitivity_scores
                ],
            }
        else:
            return [torch.max(score) for score in sensitivity_scores], [
                torch.mean(score) for score in sensitivity_scores
            ]
    else:
        if return_dict:
            return {
                "sensitivity_max": torch.max(sensitivity_scores, dim=1).values,
                "sensitivity_avg": torch.mean(sensitivity_scores, dim=1),
            }
        else:
            return (
                torch.max(sensitivity_scores, dim=1).values,
                torch.mean(sensitivity_scores, dim=1),
            )
