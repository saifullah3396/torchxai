from __future__ import annotations

from torch import nn

from torchxai.explanation_framework.explainers._grad.deeplift import DeepLiftExplainer
from torchxai.explanation_framework.explainers._grad.deeplift_shap import (
    DeepLiftShapExplainer,
)
from torchxai.explanation_framework.explainers._grad.gradient_shap import (
    GradientShapExplainer,
)
from torchxai.explanation_framework.explainers._grad.guided_backprop import (
    GuidedBackpropExplainer,
)
from torchxai.explanation_framework.explainers._grad.input_x_gradient import (
    InputXGradientExplainer,
)
from torchxai.explanation_framework.explainers._grad.integrated_gradients import (
    IntegratedGradientsExplainer,
)
from torchxai.explanation_framework.explainers._grad.saliency import SaliencyExplainer
from torchxai.explanation_framework.explainers._perturbation.feature_ablation import (
    FeatureAblationExplainer,
)
from torchxai.explanation_framework.explainers._perturbation.kernel_shap import (
    KernelShapExplainer,
)
from torchxai.explanation_framework.explainers._perturbation.lime import LimeExplainer
from torchxai.explanation_framework.explainers._perturbation.occlusion import (
    OcclusionExplainer,
)
from torchxai.explanation_framework.explainers._perturbation.shapley_value_sampling import (
    ShapleyValueSamplingExplainer,
)
from torchxai.explanation_framework.explainers.random import RandomExplainer
from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)

explainers = {
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
    "shapley_values_sampling": ShapleyValueSamplingExplainer,
}


class ExplainerFactory:
    @staticmethod
    def create(explanation_method: str, model: nn.Module, **kwargs) -> FusionExplainer:
        """
        Creates an explainer object based on the given explanation method.
        Args:
            explanation_method (str): The explanation method to be used.
        Returns:
            CaptumExplainerBase: The created CaptumExplainerBase object.
        Raises:
            ValueError: If the given explanation method is not supported.
        """

        explainer_class = explainers.get(explanation_method, None)
        if explainer_class is None:
            raise ValueError(
                f"Attribution method [{explanation_method}] is not supported. Supported methods are: {explainers.keys()}."
            )
        return explainer_class(model, **kwargs)
