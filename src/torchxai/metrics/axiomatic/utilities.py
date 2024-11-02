from copy import deepcopy
from functools import reduce
from typing import Any

import torch
from captum._utils.common import _format_additional_forward_args
from torch import nn


def _create_model_with_shifted_bias(
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
        forward_hooks = []
        saved_outputs = {}
        for input_layer_name in input_layer_names:
            module = reduce(getattr, [shifted_model, *input_layer_name.split(".")])

            def output_saver(saved_outputs):
                def hook(module, input, output):
                    saved_outputs[module] = output

                return hook

            hook = module.register_forward_hook(output_saver(saved_outputs))
            forward_hooks.append(hook)

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
                    module_output[0][
                        :, module_output.shape[1] // 2, module_output.shape[1] // 2
                    ]
                )
            elif isinstance(module, nn.Linear):
                module.bias = nn.Parameter(module_output[0])
            else:
                raise ValueError("Input layer is not a Conv2d or Linear layer")

        for hook in forward_hooks:
            hook.remove()

        return shifted_model


def _create_shifted_expainer(explainer, input_layer_names, constant_shifts, **kwargs):
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
    shifted_model = _create_model_with_shifted_bias(
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
