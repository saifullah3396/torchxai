from typing import Any

from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, DeepLift

from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)


class DeepLiftExplainer(FusionExplainer):
    """
    A Explainer class for handling DeepLIFT attribution using the Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        internal_batch_size (int, optional): The batch size for internal computations. Default is 16.
    """

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        return DeepLift(self.model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the DeepLIFT attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            baselines (BaselineType): Baselines for computing attributions.
            additional_forward_args (Any): Additional arguments to the forward function.
            return_convergence_delta (bool, optional): Whether to return the convergence delta. Default is False.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        attributions, convergence_delta = self.explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
        )

        if return_convergence_delta:
            return attributions, convergence_delta
        return attributions
