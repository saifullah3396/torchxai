from typing import Any, Callable, Union

from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, IntegratedGradients
from torch.nn import Module
from torch.nn.modules import Module

from torchxai.explanation_framework.core.explainers.torch_fusion_explainer import (
    FusionExplainer,
)


class IntegratedGradientsExplainer(FusionExplainer):
    """
    A Explainer class for handling integrated gradients attribution using Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        internal_batch_size (int, optional): The batch size for internal computations. Default is 16.
        n_steps (int, optional): The number of steps for the integrated gradients approximation. Default is 100.

    Attributes:
        n_steps (int): The number of steps for integrated gradients.
    """

    def __init__(
        self,
        model: Union[Module, Callable],
        internal_batch_size: int = 1,
        n_steps: int = 200,
    ) -> None:
        """
        Initialize the IntegratedGradientsExplainer with the model, internal batch size, and steps.
        """
        super().__init__(model, internal_batch_size)
        self.n_steps = n_steps

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """

        return IntegratedGradients(self.model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType,
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
        attributions, convergence_delta = self.explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=True,
            internal_batch_size=self.internal_batch_size,
            n_steps=self.n_steps,
        )

        if return_convergence_delta:
            return attributions, convergence_delta
        return attributions
