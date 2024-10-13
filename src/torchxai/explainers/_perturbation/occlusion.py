from typing import Any, Callable, Tuple, Union

import numpy as np
import torch
from captum._utils.common import _format_tensor_into_tuples
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, Occlusion
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._utils.common import (
    _format_and_verify_sliding_window_shapes,
    _format_and_verify_strides,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn.modules import Module

from torchxai.explainers._perturbation.feature_ablation import (
    MultiTargetFeatureAblation,
)
from torchxai.explainers.explainer import Explainer


class MultiTargetOcclusion(MultiTargetFeatureAblation):
    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        FeatureAblation.__init__(self, forward_func)
        self.use_weights = True

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        sliding_window_shapes: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        strides: Union[
            None, int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]
        ] = None,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        formatted_inputs = _format_tensor_into_tuples(inputs)

        # Formatting strides
        strides = _format_and_verify_strides(strides, formatted_inputs)

        # Formatting sliding window shapes
        sliding_window_shapes = _format_and_verify_sliding_window_shapes(
            sliding_window_shapes, formatted_inputs
        )

        # Construct tensors from sliding window shapes
        sliding_window_tensors = tuple(
            torch.ones(window_shape, device=formatted_inputs[i].device)
            for i, window_shape in enumerate(sliding_window_shapes)
        )

        # Construct counts, defining number of steps to make of occlusion block in
        # each dimension.
        shift_counts = []
        for i, inp in enumerate(formatted_inputs):
            current_shape = np.subtract(inp.shape[1:], sliding_window_shapes[i])
            # Verify sliding window doesn't exceed input dimensions.
            assert (np.array(current_shape) >= 0).all(), (
                "Sliding window dimensions {} cannot exceed input dimensions" "{}."
            ).format(sliding_window_shapes[i], tuple(inp.shape[1:]))
            # Stride cannot be larger than sliding window for any dimension where
            # the sliding window doesn't cover the entire input.
            assert np.logical_or(
                np.array(current_shape) == 0,
                np.array(strides[i]) <= sliding_window_shapes[i],
            ).all(), (
                "Stride dimension {} cannot be larger than sliding window "
                "shape dimension {}."
            ).format(
                strides[i], sliding_window_shapes[i]
            )
            shift_counts.append(
                tuple(
                    np.add(np.ceil(np.divide(current_shape, strides[i])).astype(int), 1)
                )
            )

        # Use ablation attribute method
        return super().attribute.__wrapped__(
            self,
            inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            sliding_window_tensors=sliding_window_tensors,
            shift_counts=tuple(shift_counts),
            strides=strides,
            show_progress=show_progress,
        )

    def _construct_ablated_input(
        self,
        expanded_input: Tensor,
        input_mask: Union[None, Tensor],
        baseline: Union[Tensor, int, float],
        start_feature: int,
        end_feature: int,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Ablates given expanded_input tensor with given feature mask, feature range,
        and baselines, and any additional arguments.
        expanded_input shape is (num_features, num_examples, ...)
        with remaining dimensions corresponding to remaining original tensor
        dimensions and num_features = end_feature - start_feature.

        input_mask is None for occlusion, and the mask is constructed
        using sliding_window_tensors, strides, and shift counts, which are provided in
        kwargs. baseline is expected to
        be broadcastable to match expanded_input.

        This method returns the ablated input tensor, which has the same
        dimensionality as expanded_input as well as the corresponding mask with
        either the same dimensionality as expanded_input or second dimension
        being 1. This mask contains 1s in locations which have been ablated (and
        thus counted towards ablations for that feature) and 0s otherwise.
        """
        input_mask = (
            torch.stack(
                [
                    self._occlusion_mask(
                        expanded_input,
                        j,
                        kwargs["sliding_window_tensors"],
                        kwargs["strides"],
                        kwargs["shift_counts"],
                    )
                    for j in range(start_feature, end_feature)
                ],
                dim=0,
            )
            .long()
            .repeat(
                1, expanded_input.shape[1], 1, 1, 1
            )  # changes made here for multi-target
        )
        ablated_tensor = (
            expanded_input
            * (
                torch.ones(1, dtype=torch.long, device=expanded_input.device)
                - input_mask
            ).to(expanded_input.dtype)
        ) + (baseline * input_mask.to(expanded_input.dtype))
        return ablated_tensor, input_mask

    def _occlusion_mask(
        self,
        expanded_input: Tensor,
        ablated_feature_num: int,
        sliding_window_tsr: Tensor,
        strides: Union[int, Tuple[int, ...]],
        shift_counts: Tuple[int, ...],
    ) -> Tensor:
        """
        This constructs the current occlusion mask, which is the appropriate
        shift of the sliding window tensor based on the ablated feature number.
        The feature number ranges between 0 and the product of the shift counts
        (# of times the sliding window should be shifted in each dimension).

        First, the ablated feature number is converted to the number of steps in
        each dimension from the origin, based on shift counts. This procedure
        is similar to a base conversion, with the position values equal to shift
        counts. The feature number is first taken modulo shift_counts[0] to
        get the number of shifts in the first dimension (each shift
        by shift_count[0]), and then divided by shift_count[0].
        The procedure is then continued for each element of shift_count. This
        computes the total shift in each direction for the sliding window.

        We then need to compute the padding required after the window in each
        dimension, which is equal to the total input dimension minus the sliding
        window dimension minus the (left) shift amount. We construct the
        array pad_values which contains the left and right pad values for each
        dimension, in reverse order of dimensions, starting from the last one.

        Once these padding values are computed, we pad the sliding window tensor
        of 1s with 0s appropriately, which is the corresponding mask,
        and the result will match the input shape.
        """
        remaining_total = ablated_feature_num
        current_index = []
        for i, shift_count in enumerate(shift_counts):
            stride = strides[i] if isinstance(strides, tuple) else strides
            current_index.append((remaining_total % shift_count) * stride)
            remaining_total = remaining_total // shift_count

        remaining_padding = np.subtract(
            expanded_input.shape[2:], np.add(current_index, sliding_window_tsr.shape)
        )
        pad_values = [
            val for pair in zip(remaining_padding, current_index) for val in pair
        ]
        pad_values.reverse()
        padded_tensor = torch.nn.functional.pad(
            sliding_window_tsr, tuple(pad_values)  # type: ignore
        )
        return padded_tensor.reshape((1,) + padded_tensor.shape)

    def _get_feature_range_and_mask(
        self, input: Tensor, input_mask: Tensor, **kwargs: Any
    ) -> Tuple[int, int, None]:
        feature_max = np.prod(kwargs["shift_counts"])
        return 0, feature_max, None

    def _get_feature_counts(self, inputs, feature_mask, **kwargs):
        """return the numbers of possible input features"""
        return tuple(np.prod(counts).astype(int) for counts in kwargs["shift_counts"])


class OcclusionExplainer(Explainer):
    """
    A Explainer class for Feature Ablation using Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        perturbations_per_eval (int, optional): The number of feature perturbations evaluated per batch. Default is 200.

    Attributes:
        attr_class (Occlusion): The class representing the Feature Ablation method.
        perturbations_per_eval (int): Number of feature perturbations per evaluation.
    """

    def __init__(
        self,
        model: Union[Module, Callable],
        sliding_window_shapes: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        strides: Union[
            None, int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]
        ],
        is_multi_target: bool = False,
        internal_batch_size: int = 1,
    ) -> None:
        super().__init__(model, is_multi_target, internal_batch_size)
        self._sliding_window_shapes = sliding_window_shapes
        self._strides = strides

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        if self._is_multi_target:
            return MultiTargetOcclusion(self._model)
        return Occlusion(self._model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute Feature Ablation attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            additional_forward_args (Any): Additional arguments to forward function.
            baselines (BaselineType): Baselines for computing attributions.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """

        return self._explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            sliding_window_shapes=self._sliding_window_shapes,
            strides=self._strides,
            perturbations_per_eval=self._internal_batch_size,
            show_progress=False,
        )
