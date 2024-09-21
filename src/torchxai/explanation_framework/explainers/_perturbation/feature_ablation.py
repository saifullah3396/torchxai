from typing import Any, Tuple, Union

from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, FeatureAblation
from torch import Tensor
from torch.nn import Module

from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.explanation_framework.utils.common import _expand_feature_mask_to_target


class FeatureAblationExplainer(FusionExplainer):
    """
    A Explainer class for Feature Ablation using Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        perturbations_per_eval (int, optional): The number of feature perturbations evaluated per batch. Default is 200.

    Attributes:
        attr_class (FeatureAblation): The class representing the Feature Ablation method.
        perturbations_per_eval (int): Number of feature perturbations per evaluation.
    """

    def __init__(
        self,
        model: Module,
        perturbations_per_eval: int = 1,
    ) -> None:
        """
        Initialize the FeatureAblationExplainer with the model and perturbations per evaluation.
        """
        super().__init__(model)
        self.perturbations_per_eval = perturbations_per_eval

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """

        return FeatureAblation(self.model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute Feature Ablation attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            additional_forward_args (Any): Additional arguments to forward function.
            baselines (BaselineType): Baselines for computing attributions.
            feature_mask (Union[None, Tensor, Tuple[Tensor, ...]], optional): Masks representing feature groups.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        # Compute the attributions using Kernel SHAP
        feature_mask = _expand_feature_mask_to_target(feature_mask, inputs)

        return self.explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=self.perturbations_per_eval,
            show_progress=True,
        )
