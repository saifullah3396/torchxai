from typing import Any, Callable, Tuple, Union

import numpy as np
import torch
from captum._utils.common import _format_tensor_into_tuples
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
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

#!/usr/bin/env python3
from typing import Any, Callable, Tuple, Union

import numpy as np
import torch
from captum._utils.common import _format_tensor_into_tuples
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._utils.common import (
    _format_and_verify_sliding_window_shapes,
    _format_and_verify_strides,
)
from torch import Tensor
import math
from typing import Any, Callable, cast, Tuple, Union

import torch
from captum._utils.common import (
    _format_additional_forward_args,
    _format_feature_mask,
    _format_output,
    _is_tuple,
)
from captum._utils.progress import progress
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.common import _format_input_baseline
from torch import dtype, Tensor


class Occlusion(FeatureAblation):
    """
    This implementation is exactly as Captum except for how the weights are multiplied. We weight each output
    with respect to the total number of elements in each feature group instead of just the number of overlaps
    """

    def __init__(self, forward_func: Callable) -> None:
        FeatureAblation.__init__(self, forward_func)
        self.use_weights = True

    def _attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TensorOrTupleOfTensorsGeneric:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs, baselines = _format_input_baseline(inputs, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        num_examples = inputs[0].shape[0]
        feature_mask = _format_feature_mask(feature_mask, inputs)

        assert (
            isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
        ), "Perturbations per evaluation must be an integer and at least 1."
        with torch.no_grad():
            if show_progress:
                feature_counts = self._get_feature_counts(
                    inputs, feature_mask, **kwargs
                )
                total_forwards = (
                    sum(
                        math.ceil(count / perturbations_per_eval)
                        for count in feature_counts
                    )
                    + 1
                )  # add 1 for the initial eval
                attr_progress = progress(
                    desc=f"{self.get_name()} attribution", total=total_forwards
                )
                attr_progress.update(0)

            # Computes initial evaluation with all features, which is compared
            # to each ablated result.
            initial_eval = self._strict_run_forward(
                self.forward_func, inputs, target, additional_forward_args
            )

            if show_progress:
                attr_progress.update()

            # number of elements in the output of forward_func
            n_outputs = initial_eval.numel() if isinstance(initial_eval, Tensor) else 1

            # flatten eval outputs into 1D (n_outputs)
            # add the leading dim for n_feature_perturbed
            flattened_initial_eval = initial_eval.reshape(1, -1)

            # Initialize attribution totals and counts
            attrib_type = cast(dtype, flattened_initial_eval.dtype)

            total_attrib = [
                # attribute w.r.t each output element
                torch.zeros(
                    (n_outputs,) + input.shape[1:],
                    dtype=attrib_type,
                    device=input.device,
                )
                for input in inputs
            ]

            # Weights are used in cases where ablations may be overlapping.
            if self.use_weights:
                weights = [
                    torch.zeros(
                        (n_outputs,) + input.shape[1:], device=input.device
                    ).float()
                    for input in inputs
                ]

            # Iterate through each feature tensor for ablation
            for i in range(len(inputs)):
                # Skip any empty input tensors
                if torch.numel(inputs[i]) == 0:
                    continue

                for (
                    current_inputs,
                    current_add_args,
                    current_target,
                    current_mask,
                ) in self._ith_input_ablation_generator(
                    i,
                    inputs,
                    additional_forward_args,
                    target,
                    baselines,
                    feature_mask,
                    perturbations_per_eval,
                    **kwargs,
                ):
                    # modified_eval has (n_feature_perturbed * n_outputs) elements
                    # shape:
                    #   agg mode: (*initial_eval.shape)
                    #   non-agg mode:
                    #     (feature_perturbed * batch_size, *initial_eval.shape[1:])
                    modified_eval = self._strict_run_forward(
                        self.forward_func,
                        current_inputs,
                        current_target,
                        current_add_args,
                    )

                    if show_progress:
                        attr_progress.update()

                    # if perturbations_per_eval > 1, the output shape must grow with
                    # input and not be aggregated
                    if perturbations_per_eval > 1 and not self._is_output_shape_valid:
                        current_batch_size = current_inputs[0].shape[0]

                        # number of perturbation, which is not the same as
                        # perturbations_per_eval when not enough features to perturb
                        n_perturb = current_batch_size / num_examples

                        current_output_shape = modified_eval.shape

                        # use initial_eval as the forward of perturbations_per_eval = 1
                        initial_output_shape = initial_eval.shape

                        assert (
                            # check if the output is not a scalar
                            current_output_shape
                            and initial_output_shape
                            # check if the output grow in same ratio, i.e., not agg
                            and current_output_shape[0]
                            == n_perturb * initial_output_shape[0]
                        ), (
                            "When perturbations_per_eval > 1, forward_func's output "
                            "should be a tensor whose 1st dim grow with the input "
                            f"batch size: when input batch size is {num_examples}, "
                            f"the output shape is {initial_output_shape}; "
                            f"when input batch size is {current_batch_size}, "
                            f"the output shape is {current_output_shape}"
                        )

                        self._is_output_shape_valid = True

                    # reshape the leading dim for n_feature_perturbed
                    # flatten each feature's eval outputs into 1D of (n_outputs)
                    modified_eval = modified_eval.reshape(-1, n_outputs)
                    # eval_diff in shape (n_feature_perturbed, n_outputs)
                    eval_diff = flattened_initial_eval - modified_eval

                    # append the shape of one input example
                    # to make it broadcastable to mask
                    eval_diff = eval_diff.reshape(
                        eval_diff.shape + (inputs[i].dim() - 1) * (1,)
                    )
                    eval_diff = eval_diff.to(total_attrib[i].device)

                    if self.use_weights:
                        weights[i] += (
                            current_mask.float().sum(dim=0)
                            * current_mask[0].float().sum()
                        )

                    total_attrib[i] += (eval_diff * current_mask.to(attrib_type)).sum(
                        dim=0
                    )

            if show_progress:
                attr_progress.close()

            # Divide total attributions by counts and return formatted attributions
            if self.use_weights:
                attrib = tuple(
                    single_attrib.float() / weight
                    for single_attrib, weight in zip(total_attrib, weights)
                )
            else:
                attrib = tuple(total_attrib)
            _result = _format_output(is_inputs_tuple, attrib)
        return _result

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
        return self._attribute(
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
        input_mask = torch.stack(
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
        ).long()
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
            show_progress=True,
        )
