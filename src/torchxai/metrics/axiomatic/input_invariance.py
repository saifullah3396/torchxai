from copy import deepcopy
from typing import Any, Tuple, Union

import torch
from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from torch import Tensor, nn

from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)


def _prepare_kwargs_for_base_and_shifted_inputs(kwargs: Any) -> Any:
    kwargs_copy = deepcopy(kwargs)
    shifted_kwargs_copy = deepcopy(kwargs)

    kwargs_copy.pop("shifted_baselines", None)
    if "shifted_baselines" in shifted_kwargs_copy:
        shifted_kwargs_copy["baselines"] = shifted_kwargs_copy.pop(
            "shifted_baselines", None
        )
    if "additional_forward_args" in kwargs_copy:
        kwargs_copy["additional_forward_args"] = _format_additional_forward_args(
            kwargs_copy["additional_forward_args"]
        )
        shifted_kwargs_copy["additional_forward_args"] = (
            _format_additional_forward_args(
                shifted_kwargs_copy["additional_forward_args"]
            )
        )
    return kwargs_copy, shifted_kwargs_copy


def create_model_with_shifted_bias(
    model,
    input_layer_names: str,
    constant_shifts: torch.Tensor,
    additional_forward_args: Any,
):
    with torch.no_grad():
        # create a copy of the model
        shifted_model = deepcopy(model)
        shifted_model.eval()

        # add hooks to save the output of the input layers
        for input_layer_name in input_layer_names:
            module = getattr(shifted_model, input_layer_name)
            saved_outputs = {}

            def output_saver(saved_outputs):
                def hook(module, input, output):
                    saved_outputs[module] = output

                return hook

            module.register_forward_hook(output_saver(saved_outputs))

        # perform a forward pass to save the input layer outputs
        shifted_model(
            *(
                (*constant_shifts, *additional_forward_args)
                if additional_forward_args is not None
                else constant_shifts
            )
        )

        # set the bias of the input layers to the saved outputs
        for module, module_output in saved_outputs.items():
            if isinstance(module, nn.Conv2d):
                # since in CNNs the bias exists as a single value for each kernel, we take the middle value of the output
                # however note that this will only work reasonably if a constant shift of the same value is applied
                # throughout the image, even then, without padding the output will be slightly shifted around the corners
                module.bias = nn.Parameter(
                    module_output[
                        :, module_output.shape[1] // 2, module_output.shape[1] // 2
                    ]
                )
            elif isinstance(module, nn.Linear):
                module.bias = nn.Parameter(module_output)
            else:
                raise ValueError("Input layer is not a Conv2d or Linear layer")
        return shifted_model


def create_shifted_expainer(explainer, input_layer_names, constant_shifts, **kwargs):
    # we need to recreate the explainer with the shifted model
    # get the model and create a shifted version of it
    model_attr = None
    possible_model_attrs = ["model", "forward_func"]
    for possible_model_attr in possible_model_attrs:
        if hasattr(explainer, possible_model_attr):
            model_attr = possible_model_attr
            break
    if model_attr is None:
        raise ValueError(
            "Explanation function must have one of the following attributes: "
            + ", ".join(possible_model_attrs)
        )
    shifted_model = create_model_with_shifted_bias(
        getattr(explainer, model_attr),
        input_layer_names=input_layer_names,
        constant_shifts=constant_shifts,
        additional_forward_args=(
            kwargs["additional_forward_args"]
            if "additional_forward_args" in kwargs
            else None
        ),
    )

    shifted_explainer = deepcopy(explainer)
    setattr(shifted_explainer, model_attr, shifted_model)

    return shifted_explainer


def input_invariance(
    explainer: Union[FusionExplainer, Attribution],
    inputs: TensorOrTupleOfTensorsGeneric,
    constant_shifts: TensorOrTupleOfTensorsGeneric,
    input_layer_names: Tuple[str],
    atol: float = 1e-8,
    **kwargs: Any,
) -> Tensor:
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

        atol (float, optional): The absolute tolerance parameter for the allclose function which checks whether
            the explanations generated for original inputs and shifted model and inptus are equal.
            Default is 1e-8.

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
        >>> input_invarance_result, inputs_expl, shifted_inputs_expl = input_invarance(saliency, input, target = 3)

    """

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

    shifted_explainer = create_shifted_expainer(
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
        if isinstance(explainer, FusionExplainer):
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
        input_invariance = sum(
            tuple(
                torch.tensor(
                    [
                        torch.allclose(
                            per_sample_input_expl,
                            per_sample_shifted_input_expl,
                            atol=atol,
                        )
                        for per_sample_input_expl, per_sample_shifted_input_expl in zip(
                            input_expl, shifted_input_expl
                        )
                    ],
                    dtype=torch.bool,
                    device=inputs[0].device,
                )
                for input_expl, shifted_input_expl in zip(
                    inputs_expl, shifted_inputs_expl
                )
            )
        ).bool()

        return (
            input_invariance,
            _format_output(is_inputs_tuple, inputs_expl),
            _format_output(is_inputs_tuple, shifted_inputs_expl),
        )
