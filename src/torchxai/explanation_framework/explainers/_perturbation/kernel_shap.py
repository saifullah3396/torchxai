from typing import Any, Tuple, Union

from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, KernelShap
from torch import Tensor
from torch.nn import Module

from torchxai.explanation_framework.explainers._utils import _generate_mask_weights
from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.explanation_framework.utils.common import _expand_feature_mask_to_target


class KernelShapExplainer(FusionExplainer):
    """
    A Explainer class for handling Kernel SHAP (SHapley Additive exPlanations) using the Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        n_samples (int, optional): The number of samples to use for Kernel SHAP. Default is 100.
        perturbations_per_eval (int, optional): The number of perturbations evaluated per batch. Default is 50.

    Attributes:
        n_samples (int): The number of samples to use for Kernel SHAP.
        perturbations_per_eval (int): The number of perturbations evaluated per batch.
    """

    def __init__(
        self,
        model: Module,
        n_samples: int = 100,
        perturbations_per_eval: int = 1,
    ) -> None:
        """
        Initialize the KernelShapExplainer with the model, number of samples, and perturbations per evaluation.
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
        return KernelShap(self.model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        additional_forward_args: Any = None,
        weight_attributions: bool = True,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the Kernel SHAP attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            baselines (BaselineType, optional): Baselines for computing attributions. Default is None.
            feature_mask (Union[None, Tensor, Tuple[Tensor, ...]], optional): Masks representing feature groups. Default is None.
            additional_forward_args (Any, optional): Additional arguments to forward to the model. Default is None.
            weight_attributions (bool, optional): Whether to weight the attributions by the feature masks. Default is True.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        # Compute the attributions using Kernel SHAP
        feature_mask = _expand_feature_mask_to_target(feature_mask, inputs)

        attributions = self.explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            n_samples=self.n_samples,
            perturbations_per_eval=self.perturbations_per_eval,
            show_progress=False,
        )

        # Optionally weight attributions by the feature mask
        if weight_attributions and feature_mask is not None:
            feature_mask_weights = tuple(
                _generate_mask_weights(x) for x in feature_mask
            )
            attributions = tuple(
                attribution * feature_mask_weight
                for attribution, feature_mask_weight in zip(
                    attributions, feature_mask_weights
                )
            )
        return attributions
