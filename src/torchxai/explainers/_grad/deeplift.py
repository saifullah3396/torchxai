import warnings
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.version
from captum._utils.common import (
    _expand_target,
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    _is_tuple,
    _run_forward,
    _select_targets,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, DeepLift
from captum.attr._core.deep_lift import SUPPORTED_NON_LINEAR, nonlinear
from captum.attr._utils.common import (
    _call_custom_attribution_func,
    _compute_conv_delta_and_format_attrs,
    _tensorize_baseline,
    _validate_input,
)
from captum.log import log_usage
from torch import Tensor, nn
from torch.nn import Module

from torchxai.explainers._utils import (
    _compute_gradients_sequential_autograd,
    _compute_gradients_vmap_autograd,
    _verify_target_for_multi_target_impl,
)
from torchxai.explainers.explainer import Explainer

# replace the softmax with nonlinear as the normalization in the softmax function is not invariant to the batch size!
# the softmax implementation results in differnt deltas for higher or lower batch sizes. Seems incorrect!
# also see https://github.com/pytorch/captum/issues/519
SUPPORTED_NON_LINEAR[nn.Softmax] = nonlinear


class MultiTargetDeepLift(DeepLift):
    def __init__(
        self,
        model: Module,
        multiply_by_inputs: bool = True,
        eps: float = 1e-10,
        gradient_func=(
            _compute_gradients_vmap_autograd
            if torch.__version__ >= "2.3.0"
            else _compute_gradients_sequential_autograd
        ),
        grad_batch_size: int = 10,
    ) -> None:
        super().__init__(model, multiply_by_inputs, eps)

        self.gradient_func = gradient_func
        self.grad_batch_size = grad_batch_size

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: Tuple[TargetType, ...] = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_tensor_into_tuples(inputs)
        baselines = _format_baseline(baselines, inputs)

        gradient_mask = apply_gradient_requirements(inputs)

        _validate_input(inputs, baselines)

        # verify that the target is valid
        _verify_target_for_multi_target_impl(inputs, target)

        # set hooks for baselines
        warnings.warn(
            """Setting forward, backward hooks and attributes on non-linear
            activations. The hooks and attributes will be removed
            after the attribution is finished"""
        )
        baselines = _tensorize_baseline(inputs, baselines)
        main_model_hooks = []
        try:
            main_model_hooks = self._hook_main_model()

            self.model.apply(self._register_hooks)

            additional_forward_args = _format_additional_forward_args(
                additional_forward_args
            )

            if isinstance(target, list):
                expanded_target = [_expand_target(t, 2) for t in target]
            else:
                expanded_target = _expand_target(target, 2)

            wrapped_forward_func = self._construct_forward_func(
                self.model,
                (inputs, baselines),
                expanded_target,
                additional_forward_args,
            )
            multi_target_gradients = self.gradient_func(
                wrapped_forward_func, inputs, grad_batch_size=self.grad_batch_size
            )

            def gradients_to_attributions(gradients):
                if custom_attribution_func is None:
                    if self.multiplies_by_inputs:
                        attributions = tuple(
                            (input - baseline) * gradient
                            for input, baseline, gradient in zip(
                                inputs, baselines, gradients
                            )
                        )
                    else:
                        attributions = gradients
                else:
                    attributions = _call_custom_attribution_func(
                        custom_attribution_func, gradients, inputs, baselines
                    )
                return attributions

            multi_target_attributions = [
                gradients_to_attributions(per_target_grad)
                for per_target_grad in multi_target_gradients
            ]
        finally:
            # Even if any error is raised, remove all hooks before raising
            self._remove_hooks(main_model_hooks)

        undo_gradient_requirements(inputs, gradient_mask)

        # computes approximation error based on the completeness axiom
        if (
            target is not None
            and isinstance(target, (list, tuple))
            and len(target) == len(multi_target_attributions)
        ):
            return [
                _compute_conv_delta_and_format_attrs(
                    self,
                    return_convergence_delta,
                    per_target_attribution,
                    baselines,
                    inputs,
                    additional_forward_args,
                    single_target,
                    is_inputs_tuple,
                )
                for single_target, per_target_attribution in zip(
                    target, multi_target_attributions
                )
            ]
        else:
            return [
                _compute_conv_delta_and_format_attrs(
                    self,
                    return_convergence_delta,
                    per_target_attribution,
                    baselines,
                    inputs,
                    additional_forward_args,
                    target,
                    is_inputs_tuple,
                )
                for per_target_attribution in multi_target_attributions
            ]

    def _construct_forward_func(
        self,
        forward_func: Callable,
        inputs: Tuple,
        target: Tuple[TargetType, ...] = None,
        additional_forward_args: Any = None,
    ) -> Callable:
        def forward_fn():
            model_out = _run_forward(
                forward_func, inputs, None, additional_forward_args
            )
            if isinstance(target, (tuple, list)):
                return torch.stack(
                    [
                        _select_targets(
                            torch.cat((model_out[:, 0], model_out[:, 1])), single_target
                        )
                        for single_target in target
                    ],
                    dim=1,
                )
            else:
                return _select_targets(
                    torch.cat((model_out[:, 0], model_out[:, 1])), target
                )

        if hasattr(forward_func, "device_ids"):
            forward_fn.device_ids = forward_func.device_ids  # type: ignore
        return forward_fn

    def _backward_hook(
        self,
        module: Module,
        grad_input: Tensor,
        grad_output: Tensor,
    ) -> Tensor:
        r"""
        `grad_input` is the gradient of the neuron with respect to its input
        `grad_output` is the gradient of the neuron with respect to its output
        we can override `grad_input` according to chain rule with.
        `grad_output` * delta_out / delta_in.

        """
        # before accessing the attributes from the module we want
        # to ensure that the properties exist, if not, then it is
        # likely that the module is being reused.
        attr_criteria = self.satisfies_attribute_criteria(module)
        if not attr_criteria:
            raise RuntimeError(
                "A Module {} was detected that does not contain some of "
                "the input/output attributes that are required for DeepLift "
                "computations. This can occur, for example, if "
                "your module is being used more than once in the network."
                "Please, ensure that module is being used only once in the "
                "network.".format(module)
            )

        multipliers = SUPPORTED_NON_LINEAR[type(module)](
            module,
            module.input,
            module.output,
            grad_input,
            grad_output,
            eps=self.eps,
        )

        # in deeplift we delete the input/output attributes but in multi-target case, we keep them as
        # we need them for computing attributions for multiple targets during autograd pass
        # del module.input
        # del module.output

        return multipliers


class DeepLiftExplainer(Explainer):
    """
    A Explainer class for handling DeepLIFT attribution using the Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        internal_batch_size (int, optional): The batch size for internal computations. Default is 16.
    """

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        if self._is_multi_target:
            return MultiTargetDeepLift(
                self._model, grad_batch_size=self._grad_batch_size
            )
        return DeepLift(self._model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: Optional[BaselineType] = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the DeepLIFT attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            baselines (BaselineType): Baselines for computing attributions.
            additional_forward_args (Any): Additional arguments to the forward function.
            return_convergence_delta (bool, optional): Whether to return the convergence delta. Default is False.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        return self._explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
        )
