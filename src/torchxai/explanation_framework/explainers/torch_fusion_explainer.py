from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, Union

import torch
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from torch import Tensor


class FusionExplainer(ABC):
    """
    Abstract base class for TorchFusion explainers.

    Attributes:
        model (Union[torch.nn.Module, Callable]): The model used for attribution computation.
        internal_batch_size (int): The internal batch size used for attribution computation.
        explanation_fn (Attribution): The attribution class used by the handler.
    """

    def __init__(
        self, model: Union[torch.nn.Module, Callable], internal_batch_size: int = 1
    ) -> None:
        self.model = model
        self.internal_batch_size = internal_batch_size
        self.explanation_fn = self._init_explanation_fn()

    @abstractmethod
    def _init_explanation_fn(self) -> Attribution:
        """
        Abstract method that returns the attribution class used by the handler.

        Returns:
            Attribution: The attribution class used by the handler.
        """

    @abstractmethod
    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Abstract method that computes the attribution for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensors for which to compute the attribution.
            target (TargetType): The target for the attribution computation.
            baselines (BaselineType, optional): The baselines for the attribution computation. Defaults to None.
            feature_mask (Union[None, Tensor, Tuple[Tensor, ...]], optional): The feature masks for the attribution computation. Defaults to None.
            additional_forward_args (Any, optional): Additional arguments to forward to the model. Defaults to None.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attribution.
        """
