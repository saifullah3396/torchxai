from typing import Any, Tuple, Union

import torch
from captum._utils.typing import (
    BaselineType,
    TargetType,
    Tensor,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr import Attribution, GradientShap
from captum.attr._core.gradient_shap import InputBaselineXGradient
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.common import _format_callable_baseline
from torch.nn.modules import Module

from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)


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


class GradientShapExplainer(FusionExplainer):
    """
    A Explainer class for GradientShap using a custom GradientShap implementation with noise tunnel.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        n_samples (int): Number of random samples used to approximate the integral.
        internal_batch_size (int): Batch size used internally for computation.
    """

    def __init__(
        self, model: Module, n_samples: int = 25, internal_batch_size: int = 1
    ) -> None:
        super().__init__(model, internal_batch_size)
        self.n_samples = n_samples

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """

        return GradientShapCustom(self.model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType,
        additional_forward_args: Any = None,
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
        return self.explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            n_samples=self.n_samples,
            n_samples_batch_size=self.internal_batch_size,
            return_convergence_delta=False,
        )
