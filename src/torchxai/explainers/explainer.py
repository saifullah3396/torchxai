from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, Union

import torch
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from torch import Tensor


class Explainer(ABC):
    """
    Abstract base class for TorchXAI explainers.

    Attributes:
        model (Union[torch.nn.Module, Callable]): The model used for attribution computation.
        is_multi_target (bool): A flag indicating whether the explainer is multi-target explainer.
        internal_batch_size (int): The internal batch size used for attribution computation.
        grad_batch_size (int): Grad batch size is used internally for batch gradient computation.
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, Callable],
        is_multi_target: bool = False,
        internal_batch_size: int = 64,
        grad_batch_size: int = 64,
    ) -> None:
        self._model = model
        self._is_multi_target = is_multi_target
        self._internal_batch_size = internal_batch_size
        self._grad_batch_size = grad_batch_size
        self._explanation_fn = self._init_explanation_fn()

    @property
    def model(self) -> Union[torch.nn.Module, Callable]:
        return self._model

    @model.setter
    def model(self, model: Union[torch.nn.Module, Callable]) -> None:
        self._model = model
        self._explanation_fn = self._init_explanation_fn()

    @model.setter
    def is_multi_target(self, is_multi_target: bool) -> None:
        self._is_multi_target = is_multi_target
        self._explanation_fn = self._init_explanation_fn()

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
