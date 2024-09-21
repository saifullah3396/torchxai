from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from ignite.utils import convert_tensor
from torchfusion.core.utilities.logging import get_logger

from torchxai.explanation_framework.batch_computation_handler.base import (
    BatchComputationHandler,
)
from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.explanation_framework.utils.common import (
    ExplanationParameters,
    unpack_explanation_parameters,
)
from torchxai.explanation_framework.utils.constants import ExplanationMetrics
from torchxai.metrics import sensitivity_max

logger = get_logger()


class SensitivityBatchComputationHandler(BatchComputationHandler):
    def __init__(
        self,
        output_file: Union[str, Path],
        perturb_radius: float = 0.02,
        n_perturb_samples: int = 50,
        max_examples_per_batch: int = 10,
    ) -> None:
        super().__init__(
            metric_name=ExplanationMetrics.SENSITIVITY.value,
            output_file=output_file,
        )
        self.perturb_radius = perturb_radius
        self.n_perturb_samples = n_perturb_samples
        self.max_examples_per_batch = max_examples_per_batch

    def _compute_metric(
        self,
        explainer: FusionExplainer,
        explanation_parameters: ExplanationParameters,
        batch_target_labels: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[List[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]]:
        (
            inputs,
            baselines,
            feature_mask,
            additional_forward_args,
        ) = unpack_explanation_parameters(explanation_parameters)

        assert (
            inputs[0].shape[0] == batch_target_labels.shape[0]
            and inputs[0].shape[0] == baselines[0].shape[0]
        ), "Input, baselines and target labels shape must match"

        # Pass necessary arguments based on the explanation method's requirements
        args = dict(
            inputs=inputs,
            target=batch_target_labels,
            additional_forward_args=additional_forward_args,
        )

        fn_parameters = inspect.signature(explainer.explain).parameters
        if "feature_mask" in fn_parameters:
            args["feature_mask"] = convert_tensor(
                feature_mask, device=batch_target_labels.device
            )
        if "baselines" in fn_parameters:
            args["baselines"] = baselines
        if "return_convergence_delta" in fn_parameters:
            args["return_convergence_delta"] = False

        for k, v in args.items():
            if isinstance(v, tuple):
                for i, tensor in enumerate(v):
                    logger.debug(f"Key: {k}[{i}], Value: {tensor.shape}")
            elif isinstance(v, torch.Tensor):
                logger.debug(f"Key: {k}, Value: {v.shape}")
            else:
                logger.debug(f"Key: {k}, Value: {v}")

        # compute metrics
        sensitivity_scores = (
            sensitivity_max(
                explainer.explain,
                **args,
                perturb_radius=self.perturb_radius,  # perturb each input by 0.02
                n_perturb_samples=self.n_perturb_samples,  # compute sensitivity for only 50 perturbations because it is too expensive otherwise
                max_examples_per_batch=self.max_examples_per_batch,
            )
            .detach()
            .cpu()
            .numpy()
        )

        return {self._metric_name: sensitivity_scores}

    def _save_outputs(
        self,
        sample_keys: List[str],
        outputs: dict[Any],
        time_taken: float,
    ) -> None:
        super()._save_outputs(sample_keys, outputs, time_taken)

        # save sensitivity scores in output file
        for sample_index, sample_key in enumerate(sample_keys):
            # save sensitivity results
            self._hfio.save(
                self._metric_name,
                outputs[self._metric_name][sample_index],
                sample_key,
            )

            # save sensitivity setup attributes
            self._hfio.save_attribute("perturb_radius", self.perturb_radius, sample_key)
            self._hfio.save_attribute(
                "n_perturb_samples", self.n_perturb_samples, sample_key
            )

    def _load_outputs(self, sample_keys: List[str]) -> Tuple[torch.Tensor | np.ndarray]:
        return {
            self._metric_name: [
                self._hfio.load(self._metric_name, sample_key)
                for sample_key in sample_keys
            ]
        }
