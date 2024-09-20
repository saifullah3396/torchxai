from typing import Any, Tuple, Union

from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, ShapleyValues
from torch import Tensor
from torch.nn import Module

from torchxai.explanation_framework.core.explainers.torch_fusion_explainer import (
    FusionExplainer,
)


class ShapleyValuesExplainer(FusionExplainer):
    """
    A Explainer class for handling Shapley Values using the Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        n_samples (int, optional): The number of samples to use for Shapley Values. Default is 200.
        perturbations_per_eval (int, optional): The number of perturbations evaluated per batch. Default is 50.

    Attributes:
        n_samples (int): The number of samples to be used for generating Shapley Values attributions.
        perturbations_per_eval (int): The number of perturbations evaluated per batch.
    """

    REQUIRES_FEATURE_MASK = True

    def __init__(
        self,
        model: Module,
        n_samples: int = 200,
        perturbations_per_eval: int = 50,
    ) -> None:
        """
        Initialize the ShapleyValuesExplainer with the model and number of samples.
        """
        super().__init__(model)
        self.n_samples = n_samples
        self.perturbations_per_eval = perturbations_per_eval

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        return ShapleyValues(self.model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType = None,
        feature_masks: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the Shapley Values attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            baselines (BaselineType, optional): Baselines for computing attributions. Default is None.
            feature_masks (Union[None, Tensor, Tuple[Tensor, ...]], optional): Masks representing feature groups. Default is None.
            additional_forward_args (Any, optional): Additional arguments to forward to the model. Default is None.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        # Compute the attributions using Shapley Values
        attributions = self.explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            feature_mask=feature_masks,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=self.perturbations_per_eval,
            show_progress=True,
        )

        return attributions
