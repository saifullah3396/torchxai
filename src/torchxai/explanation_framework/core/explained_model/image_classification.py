from __future__ import annotations

from dataclasses import is_dataclass
from typing import Mapping, Optional

import torch
from ignite.utils import convert_tensor
from torch import Tensor, nn
from torchfusion.core.constants import DataKeys

from torchxai.explanation_framework.core.explained_model.base import (  # noqa
    ExplainedModel,
    ExplanationParameters,
)
from torchxai.explanation_framework.core.utils.general import create_labelled_patch_grid


class ExplainedModelForImageClassification(ExplainedModel):
    def __init__(self, model: nn.Module, feature_grid_size: int = 16) -> None:
        super().__init__(model)
        self._feature_grid_size = feature_grid_size
        self.softmax = nn.Softmax(dim=-1)

    def _prepare_image(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if DataKeys.IMAGE not in batch:
            raise ValueError(f"Key {DataKeys.IMAGE} not found in batch")
        return batch[DataKeys.IMAGE]

    def _prepare_ref_image(self, image: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(image, dtype=torch.float)

    def prepare_explanation_parameters(
        self,
        batch: Mapping[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> ExplanationParameters:
        batch = convert_tensor(batch, device=device)
        image = self._prepare_image(batch=batch)
        ref_image = self._prepare_ref_image(image=image)
        feature_masks = create_labelled_patch_grid(
            image, grid_size=self._feature_grid_size
        )
        return ExplanationParameters(
            model_inputs=(image,),
            baselines=(ref_image,),
            feature_masks=(feature_masks,),
            additional_forward_args=(),
        )

    def output_to_labels(self, output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(output, dim=-1)

    def forward(
        self,
        image: Tensor,
    ) -> torch.Tensor:
        model_outputs = self.model(image)

        if is_dataclass(model_outputs):
            output_scores = self.softmax(model_outputs.logits)
        else:
            output_scores = self.softmax(model_outputs)

        return output_scores
