from typing import Any, Callable, List, Tuple, cast

import torch
from captum._utils.common import (
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor

from torchxai.explainers._utils import _run_forward_multi_target


def _multi_target_completeness(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions_list: List[TensorOrTupleOfTensorsGeneric],
    baselines: BaselineType,
    additional_forward_args: Any = None,
    targets_list: List[TargetType] = None,
) -> List[Tensor]:
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

        inputs = _format_tensor_into_tuples(inputs)  # type: ignore
        if baselines is None:
            baselines = tuple(torch.zeros_like(inp) for inp in inputs)
        else:
            baselines = _format_baseline(baselines, cast(Tuple[Tensor, ...], inputs))
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        attributions_list = [_format_tensor_into_tuples(attributions) for attributions in attributions_list]  # type: ignore

        # Make sure that inputs and corresponding attributions have number of tuples.
        assert len(inputs) == len(attributions_list[0]), (
            """The number of tensors in the inputs and
            attributions must match. Found number of tensors in the inputs is: {} and in the
            attributions: {}"""
        ).format(len(inputs), len(attributions_list[0]))

        # for this implementation the shapes of the inputs and attributions are not necessarily needed to be matched
        # for example the inputs can be of shape (batch_size, seq_length, n_features) and the attributions can be of shape
        # (batch_size, seq_length) where the n_features dim is summed for seq_length.

        # get the batch size
        bsz = inputs[0].size(0)

        # compute the forward pass on inputs
        inputs_fwd = _run_forward_multi_target(
            forward_func, inputs, targets_list, additional_forward_args
        )

        # compute the forward pass on baselines
        baselines_fwd = _run_forward_multi_target(
            forward_func,
            baselines,
            targets_list,
            additional_forward_args,
        )

        # compute the difference between the forward pass on inputs and baselines
        fwd_diffs = inputs_fwd - baselines_fwd

        # make fwd_diffs list
        fwd_diffs_list = [fwd_diffs[:, i] for i in range(fwd_diffs.shape[1])]

        # compute the sum of attributions
        attributions_sum_list = [
            sum(tuple(x.view(bsz, -1).sum(dim=-1) for x in attributions))
            for attributions in attributions_list
        ]

        # compute the absolute difference between the sum of attributions and the forward pass difference
        # this is the completeness score, the lower the score the better the completeness
        return [
            torch.abs(attributions_sum - fwd_diffs)
            for attributions_sum, fwd_diffs in zip(
                attributions_sum_list, fwd_diffs_list
            )
        ]
