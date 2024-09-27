from typing import Any, Callable, Tuple, Union

from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, Occlusion
from torch.nn.modules import Module

from torchxai.explainers.explainer import Explainer


class OcclusionExplainer(Explainer):
    """
    A Explainer class for Feature Ablation using Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        perturbations_per_eval (int, optional): The number of feature perturbations evaluated per batch. Default is 200.

    Attributes:
        attr_class (Occlusion): The class representing the Feature Ablation method.
        perturbations_per_eval (int): Number of feature perturbations per evaluation.
    """

    def __init__(
        self,
        model: Union[Module, Callable],
        is_multi_target: bool = False,
        internal_batch_size: int = 1,
        sliding_window_shapes: Union[
            Tuple[int, ...], Tuple[Tuple[int, ...], ...]
        ] = None,
        strides: Union[
            None, int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]
        ] = None,
    ) -> None:
        super().__init__(model, is_multi_target, internal_batch_size)
        self._sliding_window_shapes = sliding_window_shapes
        self._strides = strides

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        if self._is_multi_target:
            raise NotImplementedError(
                "Multi-target not supported for Occlusion Explainer."
            )
        return Occlusion(self._model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType = None,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute Feature Ablation attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            additional_forward_args (Any): Additional arguments to forward function.
            baselines (BaselineType): Baselines for computing attributions.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """

        return self._explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            sliding_window_shapes=self._sliding_window_shapes,
            strides=self._strides,
            perturbations_per_eval=self._internal_batch_size,
            show_progress=False,
        )
