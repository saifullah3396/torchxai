from typing import Any, Callable, Tuple

import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, Saliency
from captum.log import log_usage

from torchxai.explainers._utils import (
    _compute_gradients_sequential_autograd,
    _compute_gradients_vmap_autograd,
    _verify_target_for_multi_target_impl,
)
from torchxai.explainers.explainer import Explainer


class MultiTargetSaliency(Saliency):
    def __init__(
        self,
        forward_func: Callable,
        gradient_func=(
            _compute_gradients_vmap_autograd
            if torch.__version__ >= "2.3.0"
            else _compute_gradients_sequential_autograd
        ),
        grad_batch_size: int = 10,
    ) -> None:
        super().__init__(forward_func)
        self.gradient_func = gradient_func
        self.grad_batch_size = grad_batch_size

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: Tuple[TargetType, ...] = None,
        abs: bool = True,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_tensor_into_tuples(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        # verify that the target is valid
        _verify_target_for_multi_target_impl(inputs, target)

        # No need to format additional_forward_args here.
        # They are being formated in the `_run_forward` function in `common.py`
        multi_target_gradients = self.gradient_func(
            self.forward_func,
            inputs,
            target,
            additional_forward_args,
            grad_batch_size=self.grad_batch_size,
        )

        def gradients_to_attributions(gradients):
            if abs:
                attributions = tuple(torch.abs(gradient) for gradient in gradients)
            else:
                attributions = gradients
            return attributions

        multi_target_attributions = [
            gradients_to_attributions(per_target_grad)
            for per_target_grad in multi_target_gradients
        ]

        undo_gradient_requirements(inputs, gradient_mask)
        return [
            _format_output(is_inputs_tuple, attributions)
            for attributions in multi_target_attributions
        ]


class SaliencyExplainer(Explainer):
    """
    A Explainer class for handling saliency attribution using Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
    """

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the attribution function.

        Returns:
            Attribution: The initialized attribution function.
        """

        if self._is_multi_target:
            return MultiTargetSaliency(
                self._model, grad_batch_size=self._grad_batch_size
            )
        return Saliency(self._model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            additional_forward_args (Any): Additional arguments to forward function.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """

        return self._explanation_fn.attribute(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            abs=False,
        )
