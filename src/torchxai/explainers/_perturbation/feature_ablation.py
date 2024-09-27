import math
from typing import Any, Tuple, Union, cast

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_feature_mask,
    _format_output,
    _is_tuple,
)
from captum._utils.progress import progress
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, FeatureAblation
from captum.attr._utils.common import _format_input_baseline
from torch import Tensor, dtype

from torchxai.explainers._utils import (
    _expand_feature_mask_to_target,
    _run_forward_multi_target,
)
from torchxai.explainers.explainer import Explainer


class MultiTargetFeatureAblation(FeatureAblation):
    def attribute(
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
            # since our _strict_run_forward will always return a tensor with shape (batch_size, targets, ...)
            # the output shape is (batch_size, targets)
            output_shape = initial_eval.shape if isinstance(initial_eval, Tensor) else 1

            # flatten eval outputs into 1D (n_outputs)
            # add the leading dim for n_feature_perturbed
            flattened_initial_eval = initial_eval.reshape(1, -1)

            # Initialize attribution totals and counts
            attrib_type = cast(dtype, flattened_initial_eval.dtype)

            # for multi-target case, we generate output attributions of size batch_size * targets for each input
            total_attrib = [
                # attribute w.r.t each output element
                torch.zeros(
                    (output_shape[0] * output_shape[1],) + input.shape[1:],
                    dtype=attrib_type,
                    device=input.device,
                )
                for input in inputs
            ]

            # Weights are used in cases where ablations may be overlapping.
            if self.use_weights:
                weights = [
                    torch.zeros(
                        (output_shape[0],) + input.shape[1:], device=input.device
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
                    modified_eval = modified_eval.reshape(
                        -1, output_shape[0] * output_shape[1]
                    )

                    # eval_diff in shape (n_feature_perturbed, n_outputs)
                    eval_diff = flattened_initial_eval - modified_eval

                    # append the shape of one input example
                    # to make it broadcastable to mask
                    # at this point the eval diff looks something like this:
                    # (perturbation_steps, batch_size * n_targets)
                    eval_diff = eval_diff.reshape(
                        eval_diff.shape + (inputs[i].dim() - 1) * (1,)
                    )
                    eval_diff = eval_diff.to(total_attrib[i].device)

                    if self.use_weights:
                        weights[i] += current_mask.float().sum(dim=0)

                    # This (eval_diff * current_mask.to(attrib_type)).sum(dim=0) is of shape
                    # (perturbation_steps, batch_size * n_targets, input_shape)
                    # where each perturbation_step dimension is for a single feature
                    # so for each feature the target attribution is stored as (batch_size * n_targets, input_shape)
                    # note that for each perturbation step output, all attributions will be zero except for the
                    # feature that was perturbed in that step. In this manner the final attribution is obtained
                    # by summing over the first dimension, so all the attributions for each feature are summed
                    # independently.
                    total_attrib[i] += (
                        eval_diff
                        * current_mask.to(attrib_type).repeat(
                            (
                                1,
                                output_shape[1],
                            )  # since the current_mask is for a single feature, we repeat it for all targets
                            + (inputs[i].dim() - 1) * (1,)
                        )
                    ).sum(dim=0)
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

            attrib = tuple(
                single_attrib.reshape(
                    (output_shape[0], output_shape[1]) + single_attrib.shape[1:]
                )
                for single_attrib in attrib
            )

            attrib = [
                tuple(single_attrib[:, idx] for single_attrib in attrib)
                for idx in range(output_shape[1])
            ]

            _result = [
                _format_output(is_inputs_tuple, single_atrib) for single_atrib in attrib
            ]
        return _result

    def _strict_run_forward(self, *args, **kwargs) -> Tensor:
        """
        A temp wrapper for global _run_forward util to force forward output
        type assertion & conversion.
        Remove after the strict logic is supported by all attr classes
        """
        forward_output = _run_forward_multi_target(*args, **kwargs)
        if isinstance(forward_output, Tensor):
            if len(forward_output.shape) == 1:
                return forward_output.unsqueeze(-1)
            return forward_output

        output_type = type(forward_output)
        assert output_type is int or output_type is float, (
            "the return of forward_func must be a tensor, int, or float,"
            f" received: {forward_output}"
        )

        # using python built-in type as torch dtype
        # int -> torch.int64, float -> torch.float64
        # ref: https://github.com/pytorch/pytorch/pull/21215
        forward_output = torch.tensor(forward_output, dtype=output_type)
        if len(forward_output.shape) == 1:
            return forward_output.unsqueeze(-1)
        return forward_output

    def _ith_input_ablation_generator(
        self,
        i,
        inputs,
        additional_args,
        target,
        baselines,
        input_mask,
        perturbations_per_eval,
        **kwargs,
    ):
        """
        This method returns a generator of ablation perturbations of the i-th input

        Returns:
            ablation_iter (Generator): yields each perturbation to be evaluated
                        as a tuple (inputs, additional_forward_args, targets, mask).
        """
        extra_args = {}
        for key, value in kwargs.items():
            # For any tuple argument in kwargs, we choose index i of the tuple.
            if isinstance(value, tuple):
                extra_args[key] = value[i]
            else:
                extra_args[key] = value

        input_mask = input_mask[i] if input_mask is not None else None
        min_feature, num_features, input_mask = self._get_feature_range_and_mask(
            inputs[i], input_mask, **extra_args
        )
        num_examples = inputs[0].shape[0]
        if input_mask.shape[0] != num_examples:
            input_mask = input_mask.expand(num_examples, *input_mask.shape[1:])
        perturbations_per_eval = min(perturbations_per_eval, num_features)
        baseline = baselines[i] if isinstance(baselines, tuple) else baselines
        if isinstance(baseline, torch.Tensor):
            baseline = baseline.reshape((1,) + baseline.shape)

        if perturbations_per_eval > 1:
            # Repeat features and additional args for batch size.
            all_features_repeated = [
                torch.cat([inputs[j]] * perturbations_per_eval, dim=0)
                for j in range(len(inputs))
            ]
            additional_args_repeated = (
                _expand_additional_forward_args(additional_args, perturbations_per_eval)
                if additional_args is not None
                else None
            )
            if isinstance(target, list):
                target_repeated = [
                    _expand_target(t, perturbations_per_eval) for t in target
                ]
            else:
                target_repeated = _expand_target(target, perturbations_per_eval)
        else:
            all_features_repeated = list(inputs)
            additional_args_repeated = additional_args
            target_repeated = target

        num_features_processed = min_feature
        while num_features_processed < num_features:
            current_num_ablated_features = min(
                perturbations_per_eval, num_features - num_features_processed
            )

            # Store appropriate inputs and additional args based on batch size.
            if current_num_ablated_features != perturbations_per_eval:
                current_features = [
                    feature_repeated[0 : current_num_ablated_features * num_examples]
                    for feature_repeated in all_features_repeated
                ]
                current_additional_args = (
                    _expand_additional_forward_args(
                        additional_args, current_num_ablated_features
                    )
                    if additional_args is not None
                    else None
                )
                if isinstance(target, list):
                    current_target = [
                        _expand_target(t, current_num_ablated_features) for t in target
                    ]
                else:
                    current_target = _expand_target(
                        target, current_num_ablated_features
                    )
            else:
                current_features = all_features_repeated
                current_additional_args = additional_args_repeated
                current_target = target_repeated

            # Store existing tensor before modifying
            original_tensor = current_features[i]
            # Construct ablated batch for features in range num_features_processed
            # to num_features_processed + current_num_ablated_features and return
            # mask with same size as ablated batch. ablated_features has dimension
            # (current_num_ablated_features, num_examples, inputs[i].shape[1:])
            # Note that in the case of sparse tensors, the second dimension
            # may not necessarilly be num_examples and will match the first
            # dimension of this tensor.
            current_reshaped = current_features[i].reshape(
                (current_num_ablated_features, -1) + current_features[i].shape[1:]
            )
            ablated_features, current_mask = self._construct_ablated_input(
                current_reshaped,
                input_mask,
                baseline,
                num_features_processed,
                num_features_processed + current_num_ablated_features,
                **extra_args,
            )

            # current_features[i] has dimension
            # (current_num_ablated_features * num_examples, inputs[i].shape[1:]),
            # which can be provided to the model as input.
            current_features[i] = ablated_features.reshape(
                (-1,) + ablated_features.shape[2:]
            )
            yield tuple(
                current_features
            ), current_additional_args, current_target, current_mask
            # Replace existing tensor at index i.
            current_features[i] = original_tensor
            num_features_processed += current_num_ablated_features


class FeatureAblationExplainer(Explainer):
    """
    A Explainer class for Feature Ablation using Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        perturbations_per_eval (int, optional): The number of feature perturbations evaluated per batch. Default is 200.

    Attributes:
        attr_class (FeatureAblation): The class representing the Feature Ablation method.
        perturbations_per_eval (int): Number of feature perturbations per evaluation.
    """

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        if self._is_multi_target:
            return MultiTargetFeatureAblation(self._model)
        return FeatureAblation(self._model)

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

        return self._explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=2,
            show_progress=False,
        )
