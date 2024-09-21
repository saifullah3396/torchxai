from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch


@dataclass
class ExplanationParameters:
    model_inputs: Tuple[torch.Tensor, ...]
    baselines: Tuple[torch.Tensor, ...]
    feature_mask: Tuple[torch.Tensor, ...]
    additional_forward_args: Tuple[Any]
