from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
from captum._utils.common import _is_tuple
from captum._utils.typing import (
    BaselineType,
    TargetType,
    Tensor,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr import Attribution, GradientShap
from captum.attr._core.gradient_shap import InputBaselineXGradient, _scale_input
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.common import (
    _compute_conv_delta_and_format_attrs,
    _format_callable_baseline,
    _format_input_baseline,
)
from captum.log import log_usage
from torch.nn.modules import Module

from torchxai.explainers._grad.noise_tunnel import MultiTargetNoiseTunnel
from torchxai.explainers._utils import (
    _compute_gradients_sequential_autograd,
    _compute_gradients_vmap_autograd,
)
from torchxai.explainers.explainer import Explainer


class MultiTargetInputBaselineXGradient(InputBaselineXGradient):
    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs, baselines = _format_input_baseline(inputs, baselines)

        rand_coefficient = torch.tensor(
            np.random.uniform(0.0, 1.0, inputs[0].shape[0]),
            device=inputs[0].device,
            dtype=inputs[0].dtype,
        )

        input_baseline_scaled = tuple(
            _scale_input(input, baseline, rand_coefficient)
            for input, baseline in zip(inputs, baselines)
        )
        multi_target_gradients = self.gradient_func(
            self.forward_func,
            input_baseline_scaled,
            target,
            additional_forward_args,
            grad_batch_size=self.grad_batch_size,
        )

        def gradients_to_attributions(per_target_gradients):
            if self.multiplies_by_inputs:
                input_baseline_diffs = tuple(
                    input - baseline for input, baseline in zip(inputs, baselines)
                )
                return tuple(
                    input_baseline_diff * grad
                    for input_baseline_diff, grad in zip(
                        input_baseline_diffs, per_target_gradients
                    )
                )
            else:
                return per_target_gradients

        multi_target_attributions = [
            gradients_to_attributions(per_target_gradients)
            for per_target_gradients in multi_target_gradients
        ]

        # computes approximation error based on the completeness axiom
        if (
            target is not None
            and isinstance(target, (list, tuple))
            and len(target) == len(multi_target_attributions)
        ):
            return [
                _compute_conv_delta_and_format_attrs(
                    self,
                    return_convergence_delta,
                    per_target_attribution,
                    baselines,
                    inputs,
                    additional_forward_args,
                    single_target,
                    is_inputs_tuple,
                )
                for single_target, per_target_attribution in zip(
                    target, multi_target_attributions
                )
            ]
        else:
            return [
                _compute_conv_delta_and_format_attrs(
                    self,
                    return_convergence_delta,
                    per_target_attribution,
                    baselines,
                    inputs,
                    additional_forward_args,
                    target,
                    is_inputs_tuple,
                )
                for per_target_attribution in multi_target_attributions
            ]


class GradientShapCustom(GradientShap):
    """
    Custom implementation of GradientShap with support for batch size in noise tunnel during attribution computation.

    Args:
        inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
        baselines (Union[TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]]): Baselines.
        n_samples (int): The number of samples for the approximation.
        stdevs (Union[float, Tuple[float, ...]]): Standard deviation for noise.
        target (TargetType): Target for computing attributions.
        additional_forward_args (Any): Additional arguments for the forward function.
        return_convergence_delta (bool): Whether to return convergence delta.

    Returns:
        Union[TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]]: The computed attributions.
    """

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType,
        n_samples: int = 5,
        n_samples_batch_size: int = None,
        stdevs: Union[float, Tuple[float, ...]] = 0.0,
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        baselines = _format_callable_baseline(baselines, inputs)
        assert isinstance(
            baselines[0], torch.Tensor
        ), "Baselines distribution has to be provided in a form of a torch.Tensor {}.".format(
            baselines[0]
        )

        input_min_baseline_x_grad = InputBaselineXGradient(
            self.forward_func, self.multiplies_by_inputs
        )
        input_min_baseline_x_grad.gradient_func = self.gradient_func

        nt = NoiseTunnel(input_min_baseline_x_grad)

        attributions = nt.attribute.__wrapped__(
            nt,  # self
            inputs,
            nt_type="smoothgrad",
            nt_samples=n_samples,
            nt_samples_batch_size=n_samples_batch_size,
            stdevs=stdevs,
            draw_baseline_from_distrib=True,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
        )

        return attributions


class MultiTargetGradientShap(GradientShap):
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

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Union[
            TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]
        ],
        n_samples: int = 5,
        n_samples_batch_size: int = None,
        stdevs: Union[float, Tuple[float, ...]] = 0.0,
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        # since `baselines` is a distribution, we can generate it using a function
        # rather than passing it as an input argument
        baselines = _format_callable_baseline(baselines, inputs)
        assert isinstance(baselines[0], torch.Tensor), (
            "Baselines distribution has to be provided in a form "
            "of a torch.Tensor {}.".format(baselines[0])
        )

        input_min_baseline_x_grad = MultiTargetInputBaselineXGradient(
            self.forward_func, self.multiplies_by_inputs
        )
        input_min_baseline_x_grad.gradient_func = self.gradient_func
        input_min_baseline_x_grad.grad_batch_size = self.grad_batch_size

        nt = MultiTargetNoiseTunnel(input_min_baseline_x_grad)

        # NOTE: using attribute.__wrapped__ to not log
        attributions = nt.attribute.__wrapped__(
            nt,  # self
            inputs,
            nt_type="smoothgrad",
            nt_samples=n_samples,
            nt_samples_batch_size=n_samples_batch_size,
            stdevs=stdevs,
            target=target,
            draw_baseline_from_distrib=True,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
        )

        return attributions


class GradientShapExplainer(Explainer):
    """
    A Explainer class for GradientShap using a custom GradientShap implementation with noise tunnel.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        n_samples (int): Number of random samples used to approximate the integral.
        internal_batch_size (int): Batch size used internally for computation.
        grad_batch_size (int): Grad batch size is used internally for batch gradient computation.
    """

    def __init__(
        self,
        model: Module,
        n_samples: int = 25,
        is_multi_target: bool = False,
        internal_batch_size: int = 1,
        grad_batch_size: int = 64,
    ) -> None:
        super().__init__(
            model, is_multi_target, internal_batch_size, grad_batch_size=grad_batch_size
        )
        self.n_samples = n_samples

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        if self._is_multi_target:
            return MultiTargetGradientShap(
                self._model, grad_batch_size=self._grad_batch_size
            )
        return GradientShapCustom(self._model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: Optional[BaselineType] = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute GradientShap attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            baselines (BaselineType): Baselines for computing attributions.
            additional_forward_args (Any): Additional arguments to the forward function.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        return self._explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            n_samples=self.n_samples,
            n_samples_batch_size=self._internal_batch_size,
            return_convergence_delta=return_convergence_delta,
        )
