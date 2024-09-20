from typing import Any, Callable, Union

from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, Saliency
from torch.nn import Module

from torchxai.explanation_framework.core.explainers.torch_fusion_explainer import (
    FusionExplainer,
)


class SaliencyExplainer(FusionExplainer):
    """
    A Explainer class for handling saliency attribution using Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
    """

    def __init__(self, model: Union[Module, Callable]) -> None:
        super().__init__(model)

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the attribution function.

        Returns:
            Attribution: The initialized attribution function.
        """

        return Saliency(self.model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            additional_forward_args (Any): Additional arguments to forward function.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """

        return self.explanation_fn.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            abs=False,
        )
