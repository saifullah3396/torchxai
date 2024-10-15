from __future__ import annotations

from torch import nn

from torchxai.explainers._grad.deeplift import DeepLiftExplainer
from torchxai.explainers._grad.deeplift_shap import DeepLiftShapExplainer
from torchxai.explainers._grad.gradient_shap import GradientShapExplainer
from torchxai.explainers._grad.guided_backprop import GuidedBackpropExplainer
from torchxai.explainers._grad.input_x_gradient import InputXGradientExplainer
from torchxai.explainers._grad.integrated_gradients import IntegratedGradientsExplainer
from torchxai.explainers._grad.saliency import SaliencyExplainer
from torchxai.explainers._perturbation.feature_ablation import FeatureAblationExplainer
from torchxai.explainers._perturbation.kernel_shap import KernelShapExplainer
from torchxai.explainers._perturbation.lime import LimeExplainer
from torchxai.explainers._perturbation.occlusion import OcclusionExplainer
from torchxai.explainers.explainer import Explainer
from torchxai.explainers.random import RandomExplainer

AVAILABLE_EXPLAINERS = {
    "random": RandomExplainer,
    "saliency": SaliencyExplainer,
    "integrated_gradients": IntegratedGradientsExplainer,
    "deep_lift": DeepLiftExplainer,
    "deep_lift_shap": DeepLiftShapExplainer,
    "gradient_shap": GradientShapExplainer,
    "input_x_gradient": InputXGradientExplainer,
    "guided_backprop": GuidedBackpropExplainer,
    "feature_ablation": FeatureAblationExplainer,
    "occlusion": OcclusionExplainer,
    "lime": LimeExplainer,
    "kernel_shap": KernelShapExplainer,
}


class ExplainerFactory:
    @staticmethod
    def create(explanation_method: str, model: nn.Module, **kwargs) -> Explainer:
        """
        Creates an explainer object based on the given explanation method.
        Args:
            explanation_method (str): The explanation method to be used.
        Returns:
            CaptumExplainerBase: The created CaptumExplainerBase object.
        Raises:
            ValueError: If the given explanation method is not supported.
        """

        explainer_class = AVAILABLE_EXPLAINERS.get(explanation_method, None)
        if explainer_class is None:
            raise ValueError(
                f"Attribution method [{explanation_method}] is not supported. Supported methods are: {AVAILABLE_EXPLAINERS.keys()}."
            )
        return explainer_class(model, **kwargs)
