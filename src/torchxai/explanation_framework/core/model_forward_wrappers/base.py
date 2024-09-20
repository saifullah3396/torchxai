from __future__ import annotations

import inspect
from dataclasses import is_dataclass
from typing import Mapping, Optional

import torch
from ignite.utils import convert_tensor
from torch import nn
from torchfusion.core.constants import DataKeys

from torchxai import *  # noqa


class ModelForwardWrapper(nn.Module):
    """
    ModelForwardWrapper is a base class for mapping inputs to model and applying forward pass.
    Attributes:
        model (nn.Module): The model that converts the input data to embeddings.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()

        self.model = model
        self.softmax = nn.Softmax(dim=-1)
        self.return_outputs_only = True

    def configure(self, setup_for_explanation: bool = False) -> None:
        if setup_for_explanation:
            self.return_outputs_only = True
        else:
            self.return_outputs_only = False

    def prepare_inputs(
        self,
        batch: Mapping[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> Mapping[str, Mapping[str, torch.Tensor]]:
        return convert_tensor(batch, device=device)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Perform the forward pass of the model.
        Args:
            batch (Mapping[str, torch.Tensor]): A mapping of input tensors.
            device (Optional[torch.device], optional): The device to use for processing. Defaults to None.
        Returns:
            torch.Tensor: The output tensor of the forward pass.
        """
        model_inputs = self.prepare_inputs(batch, device=device)

        # get input parameters
        parameters = inspect.signature(self.model.forward).parameters
        if len(parameters) == 1:
            model_outputs = self.model(model_inputs[DataKeys.IMAGE])
        else:
            model_outputs = self.model(**model_inputs)

        if is_dataclass(model_outputs):
            output_scores = self.softmax(model_outputs.logits)
        else:
            output_scores = self.softmax(model_outputs)

        if self.return_outputs_only:
            return output_scores
        else:
            return model_inputs, output_scores, output_scores.argmax(dim=-1)
