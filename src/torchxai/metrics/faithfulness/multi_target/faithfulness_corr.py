import inspect
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import scipy
import torch
from captum._utils.common import (
    ExpansionTypes,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor
from torchxai.explainers._utils import _run_forward_multi_target
from torchxai.metrics._utils.batching import _divide_and_aggregate_metrics
from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _format_tensor_tuple_feature_dim,
    _validate_feature_mask,
)
from torchxai.metrics._utils.perturbation import (
    _generate_random_perturbation_masks,
    default_fixed_baseline_perturb_func,
)


def _multi_target_faithfulness_corr(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions_list: List[TensorOrTupleOfTensorsGeneric],
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    targets_list: List[TargetType] = None,
    perturb_func: Callable = default_fixed_baseline_perturb_func(),
    n_perturb_samples: int = 10,
    max_examples_per_batch: int = None,
    frozen_features: Optional[List[torch.Tensor]] = None,
    percent_features_perturbed: float = 0.1,
    show_progress: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    def _generate_perturbations(
        current_n_perturb_samples: int,
        current_n_step: int,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
        r"""
        The perturbations are generated for each example
        `current_n_perturb_samples` times.

        For performance reasons we are not calling `perturb_func` on each example but
        on a batch that contains `current_n_perturb_samples`
        repeated instances per example.
        """

        def call_perturb_func():
            r""" """
            baselines_arg = None
            inputs_arg: Union[Tensor, Tuple[Tensor, ...]]
            if len(inputs_expanded) == 1:
                inputs_arg = inputs_expanded[0]
                if baselines_expanded is not None:
                    baselines_arg = cast(Tuple, baselines_expanded)[0]
                perturbation_mask_arg = perturbation_masks[0]
            else:
                inputs_arg = inputs_expanded
                baselines_arg = baselines_expanded
                perturbation_mask_arg = perturbation_masks
            pertub_kwargs = dict(
                inputs=inputs_arg,
                perturbation_masks=perturbation_mask_arg,
            )
            if (
                inspect.signature(perturb_func).parameters.get("baselines")
                and baselines_arg is not None
            ):
                pertub_kwargs["baselines"] = baselines_arg
            return perturb_func(**pertub_kwargs)

        pert_start = current_n_step - current_n_perturb_samples
        pert_end = current_n_step
        perturbation_masks = tuple(
            torch.cat(tuple(y[pert_start:pert_end] for y in x))
            for x in global_perturbation_masks
        )

        inputs_expanded = tuple(
            torch.repeat_interleave(input, current_n_perturb_samples, dim=0)
            for input in inputs
        )

        baselines_expanded = baselines
        if baselines is not None:
            baselines_expanded = tuple(
                (
                    baseline.repeat_interleave(current_n_perturb_samples, dim=0)
                    if isinstance(baseline, torch.Tensor)
                    and baseline.shape[0] == input.shape[0]
                    else baseline
                )
                for input, baseline in zip(inputs, cast(Tuple, baselines))
            )

        return call_perturb_func(), perturbation_masks

    def _validate_inputs_and_perturbations(
        inputs: Tuple[Tensor, ...],
        inputs_perturbed: Tuple[Tensor, ...],
        perturbations: Tuple[Tensor, ...],
    ) -> None:
        # asserts the sizes of the perturbations and inputs
        assert len(inputs) == len(inputs_perturbed), (
            """The number of inputs and corresponding perturbated inputs must have the same number of
            elements. Found number of inputs is: {} and inputs_perturbed:
            {}"""
        ).format(len(inputs), len(inputs_perturbed))
        # asserts the sizes of the perturbations and inputs
        assert len(perturbations) == len(inputs), (
            """The number of perturbed
            inputs and corresponding perturbations must have the same number of
            elements. Found number of inputs is: {} and perturbations:
            {}"""
        ).format(len(perturbations), len(inputs))

    def _next_faithfulness_corr_tensors(
        current_n_perturb_samples: int,
        current_n_step: int,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]:
        inputs_perturbed, perturbation_masks = _generate_perturbations(
            current_n_perturb_samples, current_n_step
        )
        inputs_perturbed = _format_tensor_into_tuples(inputs_perturbed)
        perturbation_masks = _format_tensor_into_tuples(perturbation_masks)

        _validate_inputs_and_perturbations(
            cast(Tuple[Tensor, ...], inputs),
            cast(Tuple[Tensor, ...], inputs_perturbed),
            cast(Tuple[Tensor, ...], perturbation_masks),
        )

        targets_expanded_list = [
            _expand_target(
                target,
                current_n_perturb_samples,
                expansion_type=ExpansionTypes.repeat_interleave,
            )
            for target in targets_list
        ]
        additional_forward_args_expanded = _expand_additional_forward_args(
            additional_forward_args,
            current_n_perturb_samples,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        inputs_fwd = _run_forward_multi_target(
            forward_func, inputs, targets_list, additional_forward_args
        )
        inputs_fwd = torch.repeat_interleave(
            inputs_fwd, current_n_perturb_samples, dim=0
        )
        inputs_perturbed_fwd = _run_forward_multi_target(
            forward_func,
            inputs_perturbed,
            targets_expanded_list,
            additional_forward_args_expanded,
        )
        perturbed_fwd_diffs = inputs_fwd - inputs_perturbed_fwd
        perturbed_fwd_diffs_list = [
            perturbed_fwd_diffs[:, i] for i in range(perturbed_fwd_diffs.shape[1])
        ]
        attributions_expanded_list = [
            tuple(
                torch.repeat_interleave(attribution, current_n_perturb_samples, dim=0)
                for attribution in attributions
            )
            for attributions in attributions_list
        ]

        attributions_expanded_perturbed_sum_list = [
            sum(
                tuple(
                    (attribution * perturbation_mask)
                    .view(attributions_expanded[0].shape[0], -1)
                    .sum(dim=1)
                    for attribution, perturbation_mask in zip(
                        attributions_expanded, perturbation_masks
                    )
                )
            )
            for attributions_expanded in attributions_expanded_list
        ]

        # reshape to batch size dim and number of perturbations per example
        perturbed_fwd_diffs_list = [
            perturbed_fwd_diffs.view(bsz, -1)
            for perturbed_fwd_diffs in perturbed_fwd_diffs_list
        ]
        attributions_expanded_perturbed_sum_list = [
            attributions_expanded_perturbed_sum.view(bsz, -1)
            for attributions_expanded_perturbed_sum in attributions_expanded_perturbed_sum_list
        ]
        return perturbed_fwd_diffs_list, attributions_expanded_perturbed_sum_list

    def _agg_faithfulness_corr_tensors(agg_tensors, tensors):
        return tuple(
            [torch.cat([agg_t, t], dim=-1) for agg_t, t in zip(agg_t_list, t_list)]
            for agg_t_list, t_list in zip(agg_tensors, tensors)
        )

    with torch.no_grad():
        isinstance(
            attributions_list, list
        ), "attributions must be a list of tensors or list of tuples of tensors"
        assert isinstance(targets_list, list), "targets must be a list of targets"
        assert all(
            isinstance(x, int) for x in targets_list
        ), "targets must be a list of ints"
        assert len(targets_list) == len(attributions_list), (
            """The number of targets in the targets_list and
            attributions_list must match. Found number of targets in the targets_list is: {} and in the
            attributions_list: {}"""
        ).format(len(targets_list), len(attributions_list))
        # perform argument formattings
        inputs = _format_tensor_into_tuples(inputs)  # type: ignore
        if baselines is not None:
            baselines = _format_baseline(baselines, cast(Tuple[Tensor, ...], inputs))
            baselines = _format_tensor_tuple_feature_dim(baselines)  # type: ignore
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        attributions_list = [_format_tensor_into_tuples(attributions) for attributions in attributions_list]  # type: ignore
        feature_mask = _format_tensor_into_tuples(feature_mask)  # type: ignore

        # format feature dims for single feature dim cases
        inputs = _format_tensor_tuple_feature_dim(inputs)
        attributions_list = [
            _format_tensor_tuple_feature_dim(attributions)
            for attributions in attributions_list
        ]

        # Make sure that inputs and corresponding attributions have matching sizes.
        assert len(inputs) == len(attributions_list[0]), (
            """The number of tensors in the inputs and
            attributions must match. Found number of tensors in the inputs is: {} and in the
            attributions: {}"""
        ).format(len(inputs), len(attributions_list[0]))
        if baselines is not None:
            assert len(inputs) == len(baselines), (
                """The number of tensors in the inputs and
                baselines must match. Found number of tensors in the inputs is: {} and in the
                baselines: {}"""
            ).format(len(inputs), len(baselines))
            assert len(inputs[0]) == len(baselines[0]), (
                """The batch size in the inputs and
                baselines must match. Found batch size in the inputs is: {} and in the
                baselines: {}"""
            ).format(len(inputs[0]), len(baselines[0]))

        if feature_mask is not None:
            # assert that all elements in the feature_mask are unique and non-negative increasing
            _validate_feature_mask(feature_mask)
        else:
            # since the feature mask remains the same across targets, we can construct using just the first attribution
            feature_mask = _construct_default_feature_mask(attributions_list[0])

        # here we generate perturbation masks for the complete run in one call
        # global_perturbation_masks is a tuple of tensors, where each tensor is a perturbation mask
        # for a specific tuple input (for single inputs it is a single tensor) of shape
        # (batch_size, n_perturbations_per_sample, *input_shape)
        global_perturbation_masks = _generate_random_perturbation_masks(
            n_perturbations_per_sample=n_perturb_samples,
            feature_mask=feature_mask,
            percent_features_perturbed=percent_features_perturbed,
            frozen_features=frozen_features,
        )
        bsz = inputs[0].size(0)

        # if not normalize, directly return aggrgated MSE ((a-b)^2,)
        # else return aggregated MSE's polynomial expansion tensors (a^2, ab, b^2)
        agg_tensors = _divide_and_aggregate_metrics(
            cast(Tuple[Tensor, ...], inputs),
            n_perturb_samples,
            _next_faithfulness_corr_tensors,
            agg_func=_agg_faithfulness_corr_tensors,
            max_examples_per_batch=max_examples_per_batch,
            show_progress=show_progress,
        )
        perturbed_fwd_diffs_list = [x.detach().cpu() for x in agg_tensors[0]]
        attributions_expanded_perturbed_sum_list = [
            x.detach().cpu() for x in agg_tensors[1]
        ]

        faithfulness_corr_scores_list = [
            torch.tensor(
                [
                    scipy.stats.pearsonr(x, y)[0]
                    for x, y in zip(
                        attributions_expanded_perturbed_sum.numpy(),
                        perturbed_fwd_diffs.numpy(),
                    )
                ]
            )
            for attributions_expanded_perturbed_sum, perturbed_fwd_diffs in zip(
                attributions_expanded_perturbed_sum_list, perturbed_fwd_diffs_list
            )
        ]
    return (
        faithfulness_corr_scores_list,
        attributions_expanded_perturbed_sum_list,
        perturbed_fwd_diffs_list,
    )
