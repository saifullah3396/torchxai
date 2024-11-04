from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, IntegratedGradients
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)
from torch import Tensor
from torch.nn import Module
from torch.nn.modules import Module

from torchxai.explainers._utils import (
    _batch_attribution_multi_target,
    _compute_gradients_sequential_autograd,
    _compute_gradients_vmap_autograd,
    _verify_target_for_multi_target_impl,
)
from torchxai.explainers.explainer import Explainer


class MultiTargetIntegratedGradients(IntegratedGradients):
    def __init__(
        self,
        forward_func: Callable,
        multiply_by_inputs: bool = True,
        gradient_func=(
            _compute_gradients_vmap_autograd
            if torch.__version__ >= "2.3.0"
            else _compute_gradients_sequential_autograd
        ),
        grad_batch_size: int = 10,
    ) -> None:
        super().__init__(forward_func, multiply_by_inputs)
        self.gradient_func = gradient_func
        self.grad_batch_size = grad_batch_size

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        _validate_input(inputs, baselines, n_steps, method)

        # verify that the target is valid
        _verify_target_for_multi_target_impl(inputs, target)

        if internal_batch_size is not None:
            num_examples = inputs[0].shape[0]
            multi_target_attributions = _batch_attribution_multi_target(
                self,
                num_examples,
                internal_batch_size,
                n_steps,
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                method=method,
            )
        else:
            multi_target_attributions = self._attribute(
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
            )

        if return_convergence_delta:
            start_point, end_point = baselines, inputs
            # computes approximation error based on the completeness axiom
            if (
                target is not None
                and isinstance(target, (list, tuple))
                and len(target) == len(multi_target_attributions)
            ):
                delta = [
                    self.compute_convergence_delta(
                        per_target_attribution,
                        start_point,
                        end_point,
                        additional_forward_args=additional_forward_args,
                        target=single_target,
                    )
                    for single_target, per_target_attribution in zip(
                        target, multi_target_attributions
                    )
                ]
            else:
                delta = [
                    self.compute_convergence_delta(
                        per_target_attribution,
                        start_point,
                        end_point,
                        additional_forward_args=additional_forward_args,
                        target=target,
                    )
                    for per_target_attribution in multi_target_attributions
                ]
            return [
                _format_output(is_inputs_tuple, attributions)
                for attributions in multi_target_attributions
            ], delta
        return [
            _format_output(is_inputs_tuple, attributions)
            for attributions in multi_target_attributions
        ]

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None,
    ) -> Tuple[Tensor, ...]:
        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified
            # approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )

        if isinstance(target, list):
            expanded_target = [_expand_target(t, n_steps) for t in target]
        else:
            expanded_target = _expand_target(target, n_steps)

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        multi_target_gradients = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target=expanded_target,
            additional_forward_args=input_additional_args,
            grad_batch_size=self.grad_batch_size,
        )

        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        def gradients_to_attributions(grads):
            scaled_grads = [
                grad.contiguous().view(n_steps, -1)
                * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
                for grad in grads
            ]

            # aggregates across all steps for each tensor in the input tuple
            # total_grads has the same dimensionality as inputs
            total_grads = tuple(
                _reshape_and_sum(
                    scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
                )
                for (scaled_grad, grad) in zip(scaled_grads, grads)
            )

            # computes attribution for each tensor in input tuple
            # attributions has the same dimensionality as inputs
            if not self.multiplies_by_inputs:
                attributions = total_grads
            else:
                attributions = tuple(
                    total_grad * (input - baseline)
                    for total_grad, input, baseline in zip(
                        total_grads, inputs, baselines
                    )
                )
            return attributions

        multi_target_gradients = [
            gradients_to_attributions(grad) for grad in multi_target_gradients
        ]

        return multi_target_gradients


class IntegratedGradientsExplainer(Explainer):
    """
    A Explainer class for handling integrated gradients attribution using Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        internal_batch_size (int, optional): The batch size for internal computations. Default is 16.
        n_steps (int, optional): The number of steps for the integrated gradients approximation. Default is 100.
        grad_batch_size (int): Grad batch size is used internally for batch gradient computation.

    Attributes:
        n_steps (int): The number of steps for integrated gradients.
    """

    def __init__(
        self,
        model: Union[Module, Callable],
        is_multi_target: bool = False,
        internal_batch_size: int = 50,
        n_steps: int = 50,
        grad_batch_size: int = 64,
    ) -> None:
        """
        Initialize the IntegratedGradientsExplainer with the model, internal batch size, and steps.
        """
        super().__init__(
            model, is_multi_target, internal_batch_size, grad_batch_size=grad_batch_size
        )
        self.n_steps = n_steps

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        if self._is_multi_target:
            return MultiTargetIntegratedGradients(
                self._model, grad_batch_size=self._grad_batch_size
            )
        return IntegratedGradients(self._model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: Optional[BaselineType] = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the integrated gradients attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            baselines (BaselineType): Baselines for computing attributions.
            additional_forward_args (Any): Additional arguments to the forward function.
            return_convergence_delta (bool, optional): Whether to return the convergence delta. Default is True.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        return self._explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            internal_batch_size=self._internal_batch_size,
            n_steps=self.n_steps,
            return_convergence_delta=return_convergence_delta,
        )
