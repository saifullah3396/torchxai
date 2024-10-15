from typing import Any, Tuple

import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from torch import Tensor

from torchxai.explainers.explainer import Explainer
from torchxai.metrics.axiomatic.utilities import (
    _create_shifted_expainer,
    _prepare_kwargs_for_base_and_shifted_inputs,
)


def _multi_target_input_invariance(
    explainer: Explainer,
    inputs: TensorOrTupleOfTensorsGeneric,
    constant_shifts: TensorOrTupleOfTensorsGeneric,
    input_layer_names: Tuple[str],
    **kwargs: Any,
) -> Tensor:
    assert isinstance(
        explainer, Explainer
    ), "The explainer must be an instance of Explainer."
    assert explainer._is_multi_target, "The explainer must be a multi-target explainer."

    target = kwargs.get("target", None)
    assert isinstance(target, list), "targets must be a list of targets"
    assert all(isinstance(x, int) for x in target), "targets must be a list of ints"

    # Keeps track whether original input is a tuple or not before
    # converting it into a tuple.
    is_inputs_tuple = _is_tuple(inputs)

    kwargs_copy, shifted_kwargs_copy = _prepare_kwargs_for_base_and_shifted_inputs(
        kwargs
    )
    inputs = _format_tensor_into_tuples(inputs)  # type: ignore
    constant_shifts = _format_tensor_into_tuples(constant_shifts)  # type: ignore

    assert len(input_layer_names) == len(
        set(input_layer_names)
    ), "Each input layer must be unique for each input constant shift tensor."

    assert len(input_layer_names) == len(
        constant_shifts
    ), "The number of input layer names should be the same as the number of constant shifts. "

    assert (
        len(inputs) == len(constant_shifts)
        and inputs[0].shape[1:] == constant_shifts[0].shape
    ), (
        "The number of inputs should be the same as the number of constant shifts and the batch size of the "
        "constant shifts should be 1. "
    )

    shifted_explainer = _create_shifted_expainer(
        explainer=explainer,
        input_layer_names=input_layer_names,
        constant_shifts=constant_shifts,
        **kwargs,
    )

    # create shifted inputs
    constant_shift_expanded = tuple(
        constant_shift.unsqueeze(0).expand_as(input)
        for input, constant_shift in zip(inputs, constant_shifts)
    )
    shifted_inputs = tuple(
        input - constant_shift
        for input, constant_shift in zip(inputs, constant_shift_expanded)
    )

    with torch.no_grad():
        if isinstance(explainer, Explainer):
            inputs_expl_list = explainer.explain(inputs, **kwargs_copy)
            shifted_inputs_expl_list = shifted_explainer.explain(
                shifted_inputs, **shifted_kwargs_copy
            )
        else:
            raise ValueError(
                "Explanation function must be an instance of Attribution or FusionExplainer"
            )

        # calculate the difference between the two explanations
        input_invarance_score_list = [
            sum(
                tuple(
                    torch.tensor(
                        [
                            torch.mean(
                                torch.abs(
                                    per_sample_input_expl
                                    - per_sample_shifted_input_expl
                                )
                            ).item()
                            for per_sample_input_expl, per_sample_shifted_input_expl in zip(
                                input_expl, shifted_input_expl
                            )
                        ],
                        device=inputs[0].device,
                    )
                    for input_expl, shifted_input_expl in zip(
                        inputs_expl, shifted_inputs_expl
                    )
                )
            )
            for inputs_expl, shifted_inputs_expl in zip(
                inputs_expl_list, shifted_inputs_expl_list
            )
        ]
        return (
            input_invarance_score_list,
            [
                _format_output(is_inputs_tuple, inputs_expl)
                for inputs_expl in inputs_expl_list
            ],
            [
                _format_output(is_inputs_tuple, shifted_inputs_expl)
                for shifted_inputs_expl in shifted_inputs_expl_list
            ],
        )
