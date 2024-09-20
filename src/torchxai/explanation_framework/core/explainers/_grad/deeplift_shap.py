import typing
from typing import Any, Callable, Optional, Tuple, Union, cast

import torch
from captum._utils.common import (
    ExpansionTypes,
    _expand_additional_forward_args,
    _expand_target,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
)
from captum._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr import Attribution, DeepLift
from captum.attr._utils.common import _format_callable_baseline
from torch import Tensor
from torch.nn import Module

from torchxai.explanation_framework.core.explainers.torch_fusion_explainer import (
    FusionExplainer,
)


class DeepLiftShapBatched(DeepLift):
    def __init__(self, model: Module, multiply_by_inputs: bool = True) -> None:
        DeepLift.__init__(self, model, multiply_by_inputs=multiply_by_inputs)

    # There's a mismatch between the signatures of DeepLift.attribute and
    # DeepLiftShap.attribute, so we ignore typing here
    @typing.overload  # type: ignore
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Union[
            TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]
        ],
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: Literal[False] = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> TensorOrTupleOfTensorsGeneric: ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Union[
            TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]
        ],
        target: TargetType = None,
        additional_forward_args: Any = None,
        *,
        return_convergence_delta: Literal[True],
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]: ...

    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Union[
            TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]
        ],
        target: TargetType = None,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
        internal_batch_size: Optional[int] = None,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        baselines = _format_callable_baseline(baselines, inputs)

        assert isinstance(baselines[0], torch.Tensor) and baselines[0].shape[0] > 1, (
            "Baselines distribution has to be provided in form of a torch.Tensor"
            " with more than one example but found: {}."
            " If baselines are provided in shape of scalars or with a single"
            " baseline example, `DeepLift`"
            " approach can be used instead.".format(baselines[0])
        )

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_tensor_into_tuples(inputs)

        # batch sizes
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]

        (
            exp_inp,
            exp_base,
            exp_tgt,
            exp_addit_args,
        ) = self._expand_inputs_baselines_targets(
            baselines, inputs, target, additional_forward_args
        )

        if internal_batch_size is not None:
            num_examples = exp_inp[0].shape[0]
            agg_attributions = None
            delta = None
            for batch_idx in range(0, num_examples, internal_batch_size):
                batch_attributions = super().attribute.__wrapped__(  # type: ignore
                    self,
                    tuple(
                        x[batch_idx : batch_idx + internal_batch_size] for x in exp_inp
                    ),
                    tuple(
                        x[batch_idx : batch_idx + internal_batch_size] for x in exp_base
                    ),
                    target=exp_tgt[batch_idx : batch_idx + internal_batch_size],
                    additional_forward_args=tuple(
                        (
                            x[batch_idx : batch_idx + internal_batch_size]
                            if isinstance(x, torch.Tensor)
                            else x
                        )
                        for x in exp_addit_args
                    ),
                    return_convergence_delta=cast(
                        Literal[True, False], return_convergence_delta
                    ),
                    custom_attribution_func=custom_attribution_func,
                )
                if return_convergence_delta:
                    batch_attributions, batch_delta = cast(
                        Tuple[Tuple[Tensor, ...], Tensor], batch_attributions
                    )
                agg_attributions = (
                    tuple(
                        torch.cat(
                            (agg_attribution, batch_attribution),
                            dim=0,
                        )
                        for agg_attribution, batch_attribution in zip(
                            agg_attributions, batch_attributions
                        )
                    )
                    if agg_attributions is not None
                    else batch_attributions
                )
                delta = (
                    torch.cat((delta, batch_delta), dim=0)
                    if delta is not None
                    else batch_delta
                )
            attributions = agg_attributions
        else:
            attributions = super().attribute.__wrapped__(  # type: ignore
                self,
                exp_inp,
                exp_base,
                target=exp_tgt,
                additional_forward_args=exp_addit_args,
                return_convergence_delta=cast(
                    Literal[True, False], return_convergence_delta
                ),
                custom_attribution_func=custom_attribution_func,
            )

            if return_convergence_delta:
                attributions, delta = cast(
                    Tuple[Tuple[Tensor, ...], Tensor], attributions
                )
        attributions = tuple(
            self._compute_mean_across_baselines(
                inp_bsz, base_bsz, cast(Tensor, attribution)
            )
            for attribution in attributions
        )

        if return_convergence_delta:
            return _format_output(is_inputs_tuple, attributions), delta
        else:
            return _format_output(is_inputs_tuple, attributions)

    def _expand_inputs_baselines_targets(
        self,
        baselines: Tuple[Tensor, ...],
        inputs: Tuple[Tensor, ...],
        target: TargetType,
        additional_forward_args: Any,
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], TargetType, Any]:
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]

        expanded_inputs = tuple(
            [
                input.repeat_interleave(base_bsz, dim=0).requires_grad_()
                for input in inputs
            ]
        )
        expanded_baselines = tuple(
            [
                baseline.repeat(
                    (inp_bsz,) + tuple([1] * (len(baseline.shape) - 1))
                ).requires_grad_()
                for baseline in baselines
            ]
        )
        expanded_target = _expand_target(
            target, base_bsz, expansion_type=ExpansionTypes.repeat_interleave
        )
        input_additional_args = (
            _expand_additional_forward_args(
                additional_forward_args,
                base_bsz,
                expansion_type=ExpansionTypes.repeat_interleave,
            )
            if additional_forward_args is not None
            else None
        )
        return (
            expanded_inputs,
            expanded_baselines,
            expanded_target,
            input_additional_args,
        )

    def _compute_mean_across_baselines(
        self, inp_bsz: int, base_bsz: int, attribution: Tensor
    ) -> Tensor:
        # Average for multiple references
        attr_shape: Tuple = (inp_bsz, base_bsz)
        if len(attribution.shape) > 1:
            attr_shape += attribution.shape[1:]
        return torch.mean(attribution.view(attr_shape), dim=1, keepdim=False)


class DeepLiftShapExplainer(FusionExplainer):
    """
    A Explainer class for handling DeepLIFTSHAP attribution using the Captum library.

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
        return DeepLiftShapBatched(self.model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        train_baselines: BaselineType,
        additional_forward_args: Any = None,
        return_convergence_delta: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the DeepLIFT attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            train_baselines (BaselineType): Since deeplift shap requires baselines from the training set we explicitely
                name it as train_baselines so that we can differentiate it from the baselines used in other methods.
            additional_forward_args (Any): Additional arguments to the forward function.
            return_convergence_delta (bool, optional): Whether to return the convergence delta. Default is False.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        attributions, convergence_delta = self.explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=train_baselines,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
            internal_batch_size=self.internal_batch_size,
        )

        if return_convergence_delta:
            return attributions, convergence_delta
        return attributions
