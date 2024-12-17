from typing import Any, List, Tuple, Union

import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from torch import Tensor

from torchxai.explainers.explainer import Explainer
from torchxai.metrics.axiomatic.multi_target.input_invariance import (
    _multi_target_input_invariance,
)
from torchxai.metrics.axiomatic.utilities import (
    _create_shifted_expainer,
    _prepare_kwargs_for_base_and_shifted_inputs,
)


def _input_invariance(
    explainer: Union[Explainer, Attribution],
    inputs: TensorOrTupleOfTensorsGeneric,
    constant_shifts: TensorOrTupleOfTensorsGeneric,
    input_layer_names: Tuple[str],
    **kwargs: Any,
) -> Union[Tensor, List[Tensor]]:

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
        and inputs[0].shape[1:] == constant_shifts[0].shape[1:]
        and constant_shifts[0].shape[0] == 1
    ), (
        "The number of inputs should be the same as the number of constant shifts and the batch size of the "
        "constant shifts should be 1. Current shapes are: "
        f"{inputs[0].shape} and {constant_shifts[0].shape}"
    )

    shifted_explainer = _create_shifted_expainer(
        explainer=explainer,
        input_layer_names=input_layer_names,
        constant_shifts=constant_shifts,
        **kwargs,
    )

    # create shifted inputs
    constant_shift_expanded = tuple(
        constant_shift.expand_as(input)
        for input, constant_shift in zip(inputs, constant_shifts)
    )
    shifted_inputs = tuple(
        input - constant_shift
        for input, constant_shift in zip(inputs, constant_shift_expanded)
    )

    with torch.no_grad():
        if isinstance(explainer, Explainer):
            inputs_expl = explainer.explain(inputs, **kwargs_copy)
            shifted_inputs_expl = shifted_explainer.explain(
                shifted_inputs, **shifted_kwargs_copy
            )
        elif isinstance(explainer, Attribution):
            inputs_expl = explainer.attribute(inputs, **kwargs_copy)
            shifted_inputs_expl = shifted_explainer.attribute(
                shifted_inputs, **shifted_kwargs_copy
            )
        else:
            raise ValueError(
                "Explanation function must be an instance of Attribution or FusionExplainer"
            )

        # calculate the difference between the two explanations
        input_invarance_score = sum(
            tuple(
                torch.tensor(
                    [
                        torch.mean(
                            torch.abs(
                                per_sample_input_expl - per_sample_shifted_input_expl
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

        return input_invarance_score, inputs_expl, shifted_inputs_expl


def input_invariance(
    explainer: Union[Explainer, Attribution],
    inputs: TensorOrTupleOfTensorsGeneric,
    constant_shifts: TensorOrTupleOfTensorsGeneric,
    input_layer_names: Tuple[str],
    is_multi_target: bool = False,
    return_intermediate_results: bool = False,
    return_dict: bool = False,
    **kwargs: Any,
) -> Union[Tensor, List[Tensor]]:
    """
    Implementation of Input Invariance test by Kindermans et al., 2017. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library.

    To test for input invariance, we add a constant shift to the input data and a mean shift to the model bias,
    so that the output of the original model on the original data is equal to the output of the changed model
    on the shifted data. The metric returns True if batch attributions stayed unchanged too. Currently only
    supporting constant values for the shift.

    References:
        Pieter-Jan Kindermans et al.: "The (Un)reliability of Saliency Methods." Explainable AI (2019): 267-280

    Args:

        explainer (Union[FusionExplainer, Attribution]): The explainer instance that is used to
                compute the explanations. The explainer must be an instance of either captum Attribution class or
                the FusionExplainer instance.

        inputs (Tensor or tuple[Tensor, ...]): Input for which
                explanations are computed. If `explainer` takes a
                single tensor as input, a single input tensor should
                be provided.
                If `explainer` takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples (aka batch size), and if
                multiple input tensors are provided, the examples must
                be aligned appropriately.

        constant_shifts (Tensor or tuple[Tensor, ...]): Constant shifts defined for each input tensor.
                If `inputs` is single tensor, a single constant_shifts tensor should be provided.
                If `inputs` consists of multiple tensors, a tuple
                of the constant_shifts tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples (aka batch size), and if
                multiple input tensors are provided, the examples must
                be aligned appropriately. This constant_shift is subtracted from the input tensor and is used
                as an input baseline to shift the bias of the model in `create_model_with_shifted_bias`.
                Note that this is opposite to the original paper where the constant_shift is added to the input)

        input_layer_names (List[str]): The names of the input layers of the model that should be shifted. Each layer
                should be unique for each input constant shift tensor.

        delta (float, optional): The absolute tolerance parameter for the allclose function which checks whether
            the explanations generated for original inputs and shifted model and inptus are equal.
            Default is 1e-8.

        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class.
                Default is False.
        return_intermediate_results (bool, optional): A boolean flag that indicates whether the intermediate results
                of the metric computation are returned.
                Default is False.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
                with keys as the metric names and values as the corresponding metric outputs.
                Default is False.
        **kwargs (Any, optional): Contains a list of arguments that are passed
                to `explanation_func` explanation function which in some cases
                could be the `attribute` function of an attribution algorithm.
                Any additional arguments that need be passed to the explanation
                function should be included here.
                For instance, such arguments include:
                `additional_forward_args`, `baselines` and `target`.

    Returns:

        input_invariance (Tensor): A boolean tensor of per
            input example showing whether the explanation is invariant to the input.
            The output dimension is equal to the number of examples in the input batch.
        inputs_expl (Tensor or tuple[Tensor, ...]): The explanation for the original input.
            If `inputs` is a single tensor, a single tensor is returned.
            If `inputs` consists of multiple tensors, a tuple of tensors is returned.
        shifted_inputs_expl (Tensor or tuple[Tensor, ...]): The explanation generated for the shifted input and
            the shifted model.
            If `inputs` is a single tensor, a single tensor is returned.
            If `inputs` consists of multiple tensors, a tuple of tensors is returned.

    Examples::
        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> saliency = Saliency(net)
        >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> # Computes sensitivity score for saliency maps of class 3
        >>> input_invarance_score, inputs_expl, shifted_inputs_expl = input_invarance(saliency, input, target = 3)

    """
    metric_func = (
        _multi_target_input_invariance if is_multi_target else _input_invariance
    )
    input_invarance_score, inputs_expl, shifted_inputs_expl = metric_func(
        explainer=explainer,
        inputs=inputs,
        constant_shifts=constant_shifts,
        input_layer_names=input_layer_names,
        **kwargs,
    )

    # Keeps track whether original input is a tuple or not before
    # converting it into a tuple.
    is_inputs_tuple = _is_tuple(inputs)

    if return_intermediate_results:
        if return_dict:
            return {
                "input_invarance_score": input_invarance_score,
                "inputs_expl": _format_output(is_inputs_tuple, inputs_expl),
                "shifted_inputs_expl": _format_output(
                    is_inputs_tuple, shifted_inputs_expl
                ),
            }
        else:
            return (
                input_invarance_score,
                (
                    _format_output(is_inputs_tuple, inputs_expl)
                    if not isinstance(inputs_expl, list)
                    else inputs_expl
                ),
                (
                    _format_output(is_inputs_tuple, shifted_inputs_expl)
                    if not isinstance(shifted_inputs_expl, list)
                    else shifted_inputs_expl
                ),
            )
    else:
        if return_dict:
            return {"input_invarance_score": input_invarance_score}
        else:
            return input_invarance_score
