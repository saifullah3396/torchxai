from typing import Any

from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import LRP, Attribution

from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)


class LRPExplainer(FusionExplainer):
    """
    A Explainer class for Layer-wise Relevance Propagation (LRP) using Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
    """

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """

        return LRP(self.model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute LRP attributions for the given inputs.

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
        )
