from __future__ import annotations

from dataclasses import is_dataclass
from typing import Mapping, Optional

import torch
from ignite.utils import convert_tensor
from torch import Tensor, nn
from torchfusion.core.constants import DataKeys

from torchxai.explanation_framework.explained_model.base import (  # noqa
    ExplainedModel,
    ExplanationParameters,
)
from torchxai.explanation_framework.utils.common import grid_segmenter


class ExplainedModelForImageClassification(ExplainedModel):
    def __init__(
        self,
        model: nn.Module,
        segmentation_fn: str = "slic",
        segmentation_fn_kwargs: dict = {
            "n_segments": 100,
            "compactness": 1,
            "sigma": 1,
        },
    ) -> None:
        super().__init__(model)
        self.softmax = nn.Softmax(dim=-1)
        self.segmentation_fn = segmentation_fn
        self.segment_fn_kwargs = segmentation_fn_kwargs

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
        images = self._prepare_image(batch=batch)
        ref_images = self._prepare_ref_image(image=images)

        if self.segmentation_fn == "grid":
            feature_mask = grid_segmenter(images, **self.segment_fn_kwargs)
        elif self.segmentation_fn in ["slic"]:
            from lime.wrappers.scikit_image import SegmentationAlgorithm

            segmenter = SegmentationAlgorithm(
                self.segmentation_fn, **self.segment_fn_kwargs
            )
            feature_mask = []
            for image in images:
                feature_mask.append(
                    torch.from_numpy(segmenter(image.permute(1, 2, 0).cpu().numpy()))
                )
            feature_mask = torch.stack(feature_mask).unsqueeze(1)

        return ExplanationParameters(
            model_inputs=(images,),
            baselines=(ref_images,),
            feature_mask=(feature_mask,),
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
