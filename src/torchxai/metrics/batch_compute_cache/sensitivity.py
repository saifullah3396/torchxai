from __future__ import annotations

import inspect
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from captum.metrics import sensitivity_max
from torchfusion.core.constants import DataKeys
from torchfusion.core.utilities.logging import get_logger

from doclm.explanation_framework.core.batch_compute_cache.base import BatchComputeCache
from doclm.explanation_framework.core.explainers.torch_fusion_explainer import (
    TorchFusionExplainer,
)
from doclm.explanation_framework.core.utils.constants import EMBEDDING_KEYS
from doclm.explanation_framework.core.utils.general import unpack_inputs
from doclm.explanation_framework.core.utils.h5io import (
    HFIOMultiOutput,
    HFIOSingleOutput,
)

logger = get_logger()

SENSITIVITY_KEY = "sensitivity"


class SensitivityBatchComputeCache(BatchComputeCache):
    def __init__(
        self,
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
        perturb_radius: float = 0.02,
        n_perturb_samples: int = 50,
    ) -> None:
        super().__init__(
            metric_name=SENSITIVITY_KEY,
            hf_sample_data_io=hf_sample_data_io,
        )
        self.perturb_radius = perturb_radius
        self.n_perturb_samples = n_perturb_samples

    def compute_metric(
        self,
        explainer: TorchFusionExplainer,
        model_inputs: Tuple[Union[torch.Tensor, np.ndarray], ...],
        batch_target_labels: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[List[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]]:
        (
            inputs,
            baselines,
            feature_masks,
            extra_inputs,
            _,
        ) = unpack_inputs(model_inputs)

        assert (
            inputs[0].shape[0] == batch_target_labels.shape[0]
            and inputs[0].shape[0] == baselines[0].shape[0]
        ), "Input, baselines and target labels shape must match"

        # Pass necessary arguments based on the explanation method's requirements
        args = dict(
            inputs=inputs,
            target=batch_target_labels,
            additional_forward_args=(
                # this tuple in this order must be passed see src/doclm/models/interpretable_layoutlmv3.py
                # for forward method
                extra_inputs[EMBEDDING_KEYS.TOKEN_TYPE_EMBEDDINGS],
                extra_inputs[DataKeys.ATTENTION_MASKS],
                extra_inputs[DataKeys.TOKEN_BBOXES],
            ),
        )

        fn_parameters = inspect.signature(explainer.explain).parameters
        if "feature_masks" in fn_parameters:
            args["feature_masks"] = feature_masks
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
                max_examples_per_batch=batch_target_labels.shape[0],
            )
            .detach()
            .cpu()
            .numpy()
        )

        return {SENSITIVITY_KEY: sensitivity_scores}

    def save_outputs(
        self,
        sample_keys: List[str],
        outputs: dict[Any],
        time_taken: float,
    ) -> None:
        super().save_outputs(sample_keys, outputs, time_taken)

        # save sensitivity scores in output file
        for sample_index, sample_key in enumerate(sample_keys):
            # save sensitivity results
            self.hf_sample_data_io.save(
                SENSITIVITY_KEY,
                outputs[SENSITIVITY_KEY][sample_index],
                sample_key,
            )

            # save sensitivity setup attributes
            self.hf_sample_data_io.save_attribute(
                "perturb_radius", self.perturb_radius, sample_key
            )
            self.hf_sample_data_io.save_attribute(
                "n_perturb_samples", self.n_perturb_samples, sample_key
            )

    def load_outputs(self, sample_keys: List[str]) -> Tuple[torch.Tensor | np.ndarray]:
        return {
            SENSITIVITY_KEY: [
                self.hf_sample_data_io.load(SENSITIVITY_KEY, sample_key)
                for sample_key in sample_keys
            ]
        }
