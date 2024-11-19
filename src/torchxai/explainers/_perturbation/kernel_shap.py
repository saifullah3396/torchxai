#!/usr/bin/env python3
import itertools
from typing import Any, Callable, Generator, List, Optional, Tuple, Union

import torch
from captum._utils.common import _format_additional_forward_args
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, KernelShap
from captum.attr._core.lime import construct_feature_mask
from captum.attr._utils.common import _format_input_baseline
from captum.log import log_usage
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.nn import Module

from torchxai.explainers._perturbation.lime import MultiTargetLime, Lime
from torchxai.explainers._utils import (
    _expand_feature_mask_to_target,
    _weight_attributions,
)
from torchxai.explainers.explainer import Explainer


def kernel_shap_frozen_features_perturb_generator(
    original_inp: Union[Tensor, Tuple[Tensor, ...]], **kwargs
) -> Generator[Tensor, None, None]:
    assert "num_select_distribution" in kwargs and "num_interp_features" in kwargs, (
        "num_select_distribution and num_interp_features are necessary"
        " to use kernel_shap_perturb_func"
    )
    if isinstance(original_inp, Tensor):
        device = original_inp.device
    else:
        device = original_inp[0].device
    num_features = kwargs["num_interp_features"]
    yield torch.ones(1, num_features, device=device, dtype=torch.long)
    perturbed = torch.zeros(1, num_features, device=device, dtype=torch.long)
    if "frozen_features" in kwargs and kwargs["frozen_features"] is not None:
        perturbed[0, kwargs["frozen_features"]] = (
            1  # freeze the features, useful for padding/cls/sep tokens in sequences
        )
    yield perturbed
    while True:
        num_selected_features = kwargs["num_select_distribution"].sample()
        rand_vals = torch.randn(1, num_features)
        threshold = torch.kthvalue(
            rand_vals, num_features - num_selected_features
        ).values.item()
        perturbed = (rand_vals > threshold).to(device=device).long()
        if "frozen_features" in kwargs and kwargs["frozen_features"] is not None:
            perturbed[0, kwargs["frozen_features"]] = 1
        yield perturbed


class KernelShap(Lime):
    r"""
    Kernel SHAP is a method that uses the LIME framework to compute
    Shapley Values. Setting the loss function, weighting kernel and
    regularization terms appropriately in the LIME framework allows
    theoretically obtaining Shapley Values more efficiently than
    directly computing Shapley Values.

    More information regarding this method and proof of equivalence
    can be found in the original paper here:
    https://arxiv.org/abs/1705.07874
    """

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        Lime.__init__(
            self,
            forward_func,
            interpretable_model=SkLearnLinearRegression(),
            similarity_func=self.kernel_shap_similarity_kernel,
            perturb_func=kernel_shap_frozen_features_perturb_generator,
        )
        self.inf_weight = 1000000.0

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        frozen_features: Optional[List[int]] = None,
        return_input_shape: bool = True,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )
        num_features_list = torch.arange(num_interp_features, dtype=torch.float)
        denom = num_features_list * (num_interp_features - num_features_list)
        probs = (num_interp_features - 1) / denom
        probs[0] = 0.0
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            frozen_features=frozen_features,
            return_input_shape=return_input_shape,
            num_select_distribution=Categorical(probs),
            show_progress=show_progress,
        )

    def kernel_shap_similarity_kernel(
        self, _, __, interpretable_sample: Tensor, **kwargs
    ) -> Tensor:
        assert (
            "num_interp_features" in kwargs
        ), "Must provide num_interp_features to use default similarity kernel"
        num_selected_features = int(interpretable_sample.sum(dim=1).item())
        num_features = kwargs["num_interp_features"]
        if num_selected_features == 0 or num_selected_features == num_features:
            # weight should be theoretically infinite when
            # num_selected_features = 0 or num_features
            # enforcing that trained linear model must satisfy
            # end-point criteria. In practice, it is sufficient to
            # make this weight substantially larger so setting this
            # weight to 1000000 (all other weights are 1).
            similarities = self.inf_weight
        else:
            similarities = 1.0
        return torch.tensor([similarities])


class MultiTargetKernelShap(MultiTargetLime):

    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        Lime.__init__(
            self,
            forward_func,
            interpretable_model=SkLearnLinearRegression(),
            similarity_func=self.kernel_shap_similarity_kernel,
            perturb_func=kernel_shap_frozen_features_perturb_generator,
        )
        self.inf_weight = 1000000.0

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        frozen_features: Optional[List[int]] = None,
        return_input_shape: bool = True,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )
        num_features_list = torch.arange(num_interp_features, dtype=torch.float)
        denom = num_features_list * (num_interp_features - num_features_list)
        probs = (num_interp_features - 1) / denom
        probs[0] = 0.0
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            frozen_features=frozen_features,
            return_input_shape=return_input_shape,
            num_select_distribution=Categorical(probs),
            show_progress=show_progress,
        )

    def kernel_shap_similarity_kernel(
        self, _, __, interpretable_sample: Tensor, **kwargs
    ) -> Tensor:
        assert (
            "num_interp_features" in kwargs
        ), "Must provide num_interp_features to use default similarity kernel"
        num_selected_features = int(interpretable_sample.sum(dim=1).item())
        num_features = kwargs["num_interp_features"]
        if num_selected_features == 0 or num_selected_features == num_features:
            # weight should be theoretically infinite when
            # num_selected_features = 0 or num_features
            # enforcing that trained linear model must satisfy
            # end-point criteria. In practice, it is sufficient to
            # make this weight substantially larger so setting this
            # weight to 1000000 (all other weights are 1).
            similarities = self.inf_weight
        else:
            similarities = 1.0
        return torch.tensor([similarities])


class KernelShapExplainer(Explainer):
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
        is_multi_target: bool = False,
        internal_batch_size: int = 1,
        n_samples: int = 100,
        weight_attributions: bool = True,
    ) -> None:
        """
        Initialize the KernelShapExplainer with the model, number of samples, and perturbations per evaluation.
        """
        self._n_samples = n_samples
        self._weight_attributions = weight_attributions

        super().__init__(model, is_multi_target, internal_batch_size)

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        if self._is_multi_target:
            return MultiTargetKernelShap(self._model)
        return KernelShap(self._model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        additional_forward_args: Any = None,
        frozen_features: Optional[List[int]] = None,
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
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        attributions = self._explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            n_samples=self._n_samples,
            perturbations_per_eval=self._internal_batch_size,
            frozen_features=frozen_features,
            show_progress=True,
        )

        if self._weight_attributions and feature_mask is not None:
            if self._is_multi_target:
                attributions = [
                    _weight_attributions(attribution, feature_mask)
                    for attribution, feature_mask in zip(
                        attributions, itertools.cycle(feature_mask)
                    )
                ]
            else:
                attributions = _weight_attributions(attributions, feature_mask)
        return attributions
