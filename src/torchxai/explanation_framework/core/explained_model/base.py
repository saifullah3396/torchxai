from __future__ import annotations

from abc import abstractmethod
from typing import Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from torchxai.explanation_framework.core.utils.general import (
    ExplanationParameters,
)  # noqa


class ExplainedModel(nn.Module):
    """
    ExplainedModel is a wrapper around a model that for generating explanations on that model.
    Attributes:
        model (nn.Module): The model that converts the input data to embeddings.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()

        self.model = model
        self.return_outputs_only = True

    def configure(self, setup_for_explanation: bool = False) -> None:
        self.setup_for_explanation = setup_for_explanation

    @abstractmethod
    def prepare_explanation_parameters(
        self,
        batch: Mapping[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> ExplanationParameters:
        pass

    @abstractmethod
    def output_to_labels(self, output: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(
        self,
        model_inputs: Tuple[Tensor, ...],
    ) -> torch.Tensor:
        pass
