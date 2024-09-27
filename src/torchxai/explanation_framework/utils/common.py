from __future__ import annotations

import hashlib
from typing import Tuple

import torch

from torchxai.explainers.utils.containers import ExplanationParameters


def generate_unique_sample_key(
    sample_index: int,
    image_file_path: str,
):
    # generate unique identifier for this sample using index and image file path
    file_hash = hashlib.md5(image_file_path.encode()).hexdigest()
    sample_key = f"{sample_index}_{file_hash}"
    return sample_key


def unpack_explanation_parameters(
    explanation_parameters: ExplanationParameters,
) -> Tuple[torch.Tensor, ...]:
    if isinstance(explanation_parameters.model_inputs, dict):
        assert isinstance(explanation_parameters.baselines, dict)
        assert (
            explanation_parameters.model_inputs.keys()
            == explanation_parameters.baselines.keys()
        )
        assert (
            explanation_parameters.model_inputs.keys()
            == explanation_parameters.feature_mask.keys()
        )

        inputs = tuple(explanation_parameters.model_inputs.values())
        baselines = tuple(explanation_parameters.baselines.values())
        feature_mask = tuple(explanation_parameters.feature_mask.values())
        additional_forward_args = explanation_parameters.additional_forward_args
    else:
        inputs = explanation_parameters.model_inputs
        baselines = explanation_parameters.baselines
        feature_mask = explanation_parameters.feature_mask
        additional_forward_args = explanation_parameters.additional_forward_args

    return inputs, baselines, feature_mask, additional_forward_args


def grid_segmenter(images: torch.Tensor, cell_size: int = 16) -> torch.Tensor:
    feature_mask = []
    for image in images:
        # image dimensions are C x H x H
        dim_x, dim_y = image.shape[1] // cell_size, image.shape[2] // cell_size
        mask = (
            torch.arange(dim_x * dim_y, device=images.device)
            .view((dim_x, dim_y))
            .repeat_interleave(cell_size, dim=0)
            .repeat_interleave(cell_size, dim=1)
            .long()
            .unsqueeze(0)
        )
        feature_mask.append(mask)
    return torch.stack(feature_mask)
