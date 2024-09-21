from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from ignite.utils import convert_tensor

from torchxai.explanation_framework.batch_computation_handler.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputationHandler,
)
from torchxai.explanation_framework.utils.common import (
    _expand_feature_mask_to_target,
    unpack_explanation_parameters,
)
from torchxai.explanation_framework.utils.constants import ExplanationMetrics
from torchxai.explanation_framework.utils.containers import ExplanationParameters
from torchxai.metrics._utils.perturbation import perturb_fn_drop_batched_single_output


class InfidelityBatchComputationHandler(TorchXAIMetricBatchComputationHandler):
    def __init__(
        self,
        output_file: Union[str, Path],
        perturbation_probability: List[float] = 0.1,
        n_perturb_samples: int = 10,
        max_examples_per_batch: int = 100,
        normalize: bool = True,
    ) -> None:
        super().__init__(
            metric_name=ExplanationMetrics.INFIDELITY.value,
            output_file=output_file,
            n_perturb_samples=n_perturb_samples,
            max_examples_per_batch=max_examples_per_batch,
            normalize=normalize,
        )

        self.perturbation_probability = perturbation_probability

    def _get_metric_fn(self):
        from torchxai.metrics import infidelity

        return infidelity

    def _prepare_metric_input(
        self,
        wrapped_model: Callable,
        explanations: Tuple[Union[torch.Tensor, np.ndarray], ...],
        explanation_parameters: ExplanationParameters,
        batch_target_labels: Union[torch.Tensor, np.ndarray],
    ):
        kwargs = super()._prepare_metric_input(
            wrapped_model, explanations, explanation_parameters, batch_target_labels
        )

        (
            inputs,
            _,
            feature_mask,
            _,
        ) = unpack_explanation_parameters(explanation_parameters)

        device = batch_target_labels.device
        feature_mask = _expand_feature_mask_to_target(feature_mask, explanations)
        feature_mask = convert_tensor(feature_mask, device=device)
        kwargs["perturb_func"] = perturb_fn_drop_batched_single_output(
            feature_mask=feature_mask,
            perturbation_probability=self.perturbation_probability,
        )
        return kwargs

    def _save_sample_attributes(self, sample_key: str):
        super()._save_sample_attributes(sample_key)
        self._hfio.save_attribute(
            "perturbation_probability", self.perturbation_probability, sample_key
        )
