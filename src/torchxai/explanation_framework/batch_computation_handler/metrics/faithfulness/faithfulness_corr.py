from pathlib import Path
from typing import Callable, Tuple, Union

import numpy as np
import torch

from torchxai.explanation_framework.batch_computation_handler.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputationHandler,
)
from torchxai.explanation_framework.utils.constants import ExplanationMetrics
from torchxai.metrics._utils.perturbation import default_random_perturb_func


class FaithfulnessCorrelationBatchComputationHandler(
    TorchXAIMetricBatchComputationHandler
):
    def __init__(
        self,
        output_file: Union[str, Path],
        pertub_func_type: str = "random",
        perturbation_noise_scale: float = 0.02,
        n_perturb_samples: int = 10,
        max_examples_per_batch: int = None,
        perturbation_probability: float = 0.1,
        show_progress: bool = False,
    ) -> None:
        super().__init__(
            metric_name=ExplanationMetrics.FAITHFULNESS_CORRELATION.value,
            output_file=output_file,
            output_keys=[
                "faithfulness_corr",
                "attributions_expanded_perturbed_sum" "perturbed_fwd_diffs",
            ],
            n_perturb_samples=n_perturb_samples,
            max_examples_per_batch=max_examples_per_batch,
            perturbation_probability=perturbation_probability,
            show_progress=show_progress,
        )

        self.pertub_func_type = pertub_func_type
        self.perturbation_noise_scale = perturbation_noise_scale

        if pertub_func_type == "random":
            self.perturb_func = default_random_perturb_func(perturbation_noise_scale)
        else:
            raise ValueError(f"Invalid perturbation function type: {pertub_func_type}")

    def _get_metric_fn(self):
        from torchxai.metrics import faithfulness_corr

        return faithfulness_corr

    def _prepare_metric_input(
        self,
        wrapped_model: Callable,
        explanations: Tuple[Union[torch.Tensor, np.ndarray], ...],
        model_inputs: Tuple[Union[torch.Tensor, np.ndarray], ...],
        batch_target_labels: Union[torch.Tensor, np.ndarray],
    ):
        kwargs = super()._prepare_metric_input(
            wrapped_model, explanations, model_inputs, batch_target_labels
        )
        kwargs["perturb_func"] = self.perturb_func
        return kwargs

    def _save_sample_attributes(self, sample_key: str):
        super()._save_sample_attributes(sample_key)
        self._hfio.save_attribute("pertub_func_type", self.pertub_func_type, sample_key)
        self._hfio.save_attribute(
            "perturbation_noise_scale", self.perturbation_noise_scale, sample_key
        )
