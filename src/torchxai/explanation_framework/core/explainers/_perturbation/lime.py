from typing import Any, Tuple, Union

from captum._utils.models.linear_model import SkLearnLasso
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, Lime
from torch import Tensor
from torch.nn import Module

from torchxai.explanation_framework.core.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.explanation_framework.core.explainers.utils import generate_mask_weights
from torchxai.explanation_framework.core.utils.general import (
    expand_feature_masks_to_inputs,
)


class LimeExplainer(FusionExplainer):
    """
    A Explainer class for handling LIME (Local Interpretable Model-agnostic Explanations) using the Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        n_samples (int, optional): The number of perturbed samples to generate for LIME. Default is 100.
        perturbations_per_eval (int, optional): The number of perturbations evaluated per batch. Default is 50.

    Attributes:
        n_samples (int): The number of perturbed samples to be used for generating attributions.
        perturbations_per_eval (int): The number of perturbations evaluated per batch.
    """

    def __init__(
        self,
        model: Module,
        n_samples: int = 100,
        perturbations_per_eval: int = 10,
        alpha: float = 0.001,
    ) -> None:
        """
        Initialize the LimeExplainer with the model, number of samples, and perturbations per evaluation.
        """
        self.n_samples = n_samples
        self.perturbations_per_eval = perturbations_per_eval
        self.alpha = alpha

        super().__init__(model)

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        return Lime(self.model, interpretable_model=SkLearnLasso(alpha=self.alpha))

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType = None,
        feature_masks: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        additional_forward_args: Any = None,
        weight_attributions: bool = True,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the LIME attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            baselines (BaselineType, optional): Baselines for computing attributions. Default is None.
            feature_masks (Union[None, Tensor, Tuple[Tensor, ...]], optional): Masks representing feature groups. Default is None.
            additional_forward_args (Any, optional): Additional arguments to forward to the model. Default is None.
            weight_attributions (bool, optional): Whether to weight the attributions by the feature masks. Default is True.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        # Compute the attributions using Kernel SHAP
        feature_masks = expand_feature_masks_to_inputs(feature_masks, inputs)

        # Compute the attributions using LIME
        attributions = self.explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            feature_mask=feature_masks,
            additional_forward_args=additional_forward_args,
            n_samples=self.n_samples,
            perturbations_per_eval=self.perturbations_per_eval,
            show_progress=False,
        )

        # Optionally weight attributions by the feature mask
        if weight_attributions and feature_masks is not None:
            feature_mask_weights = tuple(
                generate_mask_weights(x) for x in feature_masks
            )
            attributions = tuple(
                attribution * feature_mask_weight
                for attribution, feature_mask_weight in zip(
                    attributions, feature_mask_weights
                )
            )

        return attributions
