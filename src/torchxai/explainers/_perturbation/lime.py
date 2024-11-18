#!/usr/bin/env python3
import itertools
import typing
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
)
from captum._utils.models.linear_model import SkLearnLasso
from captum._utils.models.model import Model
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, Lime, LimeBase
from captum.attr._core.lime import (
    _reduce_list,
    construct_feature_mask,
    default_from_interp_rep_transform,
    default_perturb_func,
    get_exp_kernel_similarity_function,
)
from captum.attr._utils.batching import _batch_example_iterator
from captum.attr._utils.common import _format_input_baseline
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module

from torchxai.explainers._perturbation.lime_base import MultiTargetLimeBase
from torchxai.explainers._utils import (
    _expand_feature_mask_to_target,
    _run_forward_multi_target,
    _weight_attributions,
)
from torchxai.explainers.explainer import Explainer


def frozen_features_perturb_func(frozen_features: Optional[List[int]]) -> Callable:
    def wrapped(original_inp, **kwargs):
        assert (
            "num_interp_features" in kwargs
        ), "Must provide num_interp_features to use default interpretable sampling function"
        if isinstance(original_inp, Tensor):
            device = original_inp.device
        else:
            device = original_inp[0].device

        probs = torch.ones(1, kwargs["num_interp_features"]) * 0.5
        perturbation = torch.bernoulli(probs).to(device=device).long()
        if frozen_features is not None:
            assert all(
                feature_idx < kwargs["num_interp_features"]
                for feature_idx in frozen_features
            ), "Frozen features must be less than num_interp_features"
            perturbation[0, frozen_features] = (
                1  # freeze the features, useful for padding/cls/sep tokens in sequences
            )
        return perturbation

    return wrapped


class MultiTargetLime(MultiTargetLimeBase):
    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Optional[Model] = None,
        similarity_func: Optional[Callable] = None,
        perturb_func: Optional[Callable] = None,
    ) -> None:
        if interpretable_model is None:
            interpretable_model = SkLearnLasso(alpha=0.01)

        if similarity_func is None:
            similarity_func = get_exp_kernel_similarity_function()

        if perturb_func is None:
            perturb_func = default_perturb_func

        LimeBase.__init__(
            self,
            forward_func,
            interpretable_model,
            similarity_func,
            perturb_func,
            True,
            default_from_interp_rep_transform,
            None,
        )

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
        return_input_shape: bool = True,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            show_progress=show_progress,
        )

    def _attribute_kwargs(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
        show_progress: bool = False,
        **kwargs,
    ) -> TensorOrTupleOfTensorsGeneric:
        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        bsz = formatted_inputs[0].shape[0]

        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )

        if num_interp_features > 10000:
            warnings.warn(
                "Attempting to construct interpretable model with > 10000 features."
                "This can be very slow or lead to OOM issues. Please provide a feature"
                "mask which groups input features to reduce the number of interpretable"
                "features. "
            )

        multi_target_coefs: Tensor
        if bsz > 1:
            test_output = _run_forward_multi_target(
                self.forward_func, inputs, target, additional_forward_args
            )

            n_targets = len(target) if isinstance(target, list) else 1
            # if the target is sent as a list of torch tensors then we need to
            if isinstance(target, list) and isinstance(target[0], Tensor):
                if target[0].shape[0] > 1:
                    assert target[0].shape[0] == bsz

                    # convert the list of tensors to multiple ids for each example
                    target = list(zip(*target))

                    target = list(item[0] for item in target)
            elif (
                isinstance(target, list)
                and isinstance(target[0], list)
                and isinstance(target[0][0], int)
            ):
                assert len(target[0]) == bsz

                # convert the list of tensors to multiple ids for each example
                target = list(zip(*target))
            elif (
                isinstance(target, list)
                and isinstance(target[0], list)
                and isinstance(target[0][0], tuple)
            ):
                assert len(target[0]) == bsz

                # convert the list of tensors to multiple ids for each example
                target = list(zip(*target))

            if isinstance(test_output, Tensor) and torch.numel(test_output) > n_targets:
                if test_output.shape[0] == bsz:
                    warnings.warn(
                        "You are providing multiple inputs for Lime / Kernel SHAP "
                        "attributions. This trains a separate interpretable model "
                        "for each example, which can be time consuming. It is "
                        "recommended to compute attributions for one example at a time."
                    )
                    output_list = []
                    for (
                        curr_inps,
                        curr_target,
                        curr_additional_args,
                        curr_baselines,
                        curr_feature_mask,
                    ) in _batch_example_iterator(
                        bsz,
                        formatted_inputs,
                        target,
                        additional_forward_args,
                        baselines,
                        feature_mask,
                    ):
                        if isinstance(curr_target, list) and isinstance(
                            curr_target[0], tuple
                        ):
                            curr_target = [[item] for item in curr_target[0]]

                        multi_target_coefs = super().attribute.__wrapped__(
                            self,
                            inputs=curr_inps if is_inputs_tuple else curr_inps[0],
                            target=curr_target,
                            additional_forward_args=curr_additional_args,
                            n_samples=n_samples,
                            perturbations_per_eval=perturbations_per_eval,
                            baselines=(
                                curr_baselines if is_inputs_tuple else curr_baselines[0]
                            ),
                            feature_mask=(
                                curr_feature_mask
                                if is_inputs_tuple
                                else curr_feature_mask[0]
                            ),
                            num_interp_features=num_interp_features,
                            show_progress=show_progress,
                            **kwargs,
                        )
                        if return_input_shape:
                            output_list.append(
                                [
                                    self._convert_output_shape(
                                        curr_inps,
                                        curr_feature_mask,
                                        coefs,
                                        num_interp_features,
                                        is_inputs_tuple,
                                    )
                                    for coefs in multi_target_coefs
                                ]
                            )
                        else:
                            output_list.append([coefs.reshape(1, -1) for coefs in multi_target_coefs])  # type: ignore

                    # switch from per sample target output to per target output
                    # each element of this output now contains the batch attributions for a single target
                    output_list = list(zip(*output_list))

                    return [_reduce_list(output) for output in output_list]
                else:
                    raise AssertionError(
                        "Invalid number of outputs, forward function should return a"
                        "scalar per example or a scalar per input batch."
                    )
            else:
                assert perturbations_per_eval == 1, (
                    "Perturbations per eval must be 1 when forward function"
                    "returns single value per batch!"
                )

        multi_target_coefs = super().attribute.__wrapped__(
            self,
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            baselines=baselines if is_inputs_tuple else baselines[0],
            feature_mask=feature_mask if is_inputs_tuple else feature_mask[0],
            num_interp_features=num_interp_features,
            show_progress=show_progress,
            **kwargs,
        )

        if return_input_shape:
            return [
                self._convert_output_shape(
                    formatted_inputs,
                    feature_mask,
                    coefs,
                    num_interp_features,
                    is_inputs_tuple,
                )
                for coefs in multi_target_coefs
            ]
        else:
            return multi_target_coefs

    @typing.overload
    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
    ) -> Tuple[Tensor, ...]: ...

    @typing.overload
    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
    ) -> Tensor: ...

    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: bool,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        coefs = coefs.flatten()
        attr = [
            torch.zeros_like(single_inp, dtype=torch.float)
            for single_inp in formatted_inp
        ]
        for tensor_ind in range(len(formatted_inp)):
            for single_feature in range(num_interp_features):
                attr[tensor_ind] += (
                    coefs[single_feature].item()
                    * (feature_mask[tensor_ind] == single_feature).float()
                )
        return _format_output(is_inputs_tuple, tuple(attr))


class LimeExplainer(Explainer):
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
        is_multi_target: bool = False,
        internal_batch_size: int = 1,
        n_samples: int = 100,
        alpha: float = 0.01,
        weight_attributions: bool = True,
    ) -> None:
        """
        Initialize the LimeExplainer with the model, number of samples, and perturbations per evaluation.
        """
        self._n_samples = n_samples
        self._alpha = alpha
        self._weight_attributions = weight_attributions

        super().__init__(model, is_multi_target, internal_batch_size)

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        if self._is_multi_target:
            return MultiTargetLime(
                self._model,
                interpretable_model=SkLearnLasso(alpha=self._alpha),
            )
        return Lime(
            self._model,
            interpretable_model=SkLearnLasso(alpha=self._alpha),
        )

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
        Compute the LIME attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            baselines (BaselineType, optional): Baselines for computing attributions. Default is None.
            feature_masks (Union[None, Tensor, Tuple[Tensor, ...]], optional): Masks representing feature groups. Default is None.
            additional_forward_args (Any, optional): Additional arguments to forward to the model. Default is None.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        # Compute the attributions using Kernel SHAP
        feature_mask = _expand_feature_mask_to_target(feature_mask, inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        # patch the perturb function to freeze features
        if frozen_features is not None:
            self._explanation_fn.perturb_func = frozen_features_perturb_func(
                frozen_features
            )

        # Compute the attributions using LIME
        attributions = self._explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            n_samples=self._n_samples,
            perturbations_per_eval=self._internal_batch_size,
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
