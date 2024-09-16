from __future__ import annotations

import inspect
from typing import Any, List, Tuple, Union

import numpy as np
import torch
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

EXPLANATIONS_KEY = "explanations"
CONVERGENCE_DELTA_KEY = "convergence_delta"


class ExplanationsBatchComputeCache(BatchComputeCache):
    def __init__(
        self,
        explanation_tuple_keys: List[str],
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
    ) -> None:
        super().__init__(
            metric_name=EXPLANATIONS_KEY,
            hf_sample_data_io=hf_sample_data_io,
        )
        self.explanation_type_keys = list(explanation_tuple_keys)

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
            input_keys,
        ) = unpack_inputs(model_inputs)

        # sanity check for input embedding names
        assert input_keys == self.explanation_type_keys, (
            f"Input keys {input_keys} must match the explanation tuple keys "
            f"{self.explanation_type_keys}"
        )

        assert (
            inputs[0].shape[0] == batch_target_labels.shape[0]
            and inputs[0].shape[0] == baselines[0].shape[0]
        ), "Input, baselines, and target labels shape must match"

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
            args["return_convergence_delta"] = True

        for k, v in args.items():
            if isinstance(v, tuple):
                for i, tensor in enumerate(v):
                    logger.debug(f"Key: {k}[{i}], Value: {tensor.shape}")
            elif isinstance(v, torch.Tensor):
                logger.debug(f"Key: {k}, Value: {v.shape}")
            else:
                logger.debug(f"Key: {k}, Value: {v}")

        outputs = explainer.explain(**args)

        if isinstance(outputs, tuple) and len(outputs) == 2:
            return {
                EXPLANATIONS_KEY: outputs[0],
                CONVERGENCE_DELTA_KEY: outputs[1],
            }
        else:
            return {
                EXPLANATIONS_KEY: outputs,
            }

    def save_outputs(
        self,
        sample_keys: List[str],
        outputs: dict[Any],
        time_taken: float,
        verify_outputs: bool = False,
    ) -> None:
        super().save_outputs(sample_keys, outputs, time_taken)

        # Save convergence delta if it's available
        if CONVERGENCE_DELTA_KEY in outputs:
            convergence_delta = outputs[CONVERGENCE_DELTA_KEY].cpu().numpy()
            for sample_index, sample_key in enumerate(sample_keys):
                self.hf_sample_data_io.save(
                    CONVERGENCE_DELTA_KEY,
                    convergence_delta[sample_index],
                    sample_key,
                )

        # Save each explanation for each sample key
        for explanation_type, explanation in zip(
            self.explanation_type_keys, outputs[EXPLANATIONS_KEY]
        ):
            for sample_index, sample_key in enumerate(sample_keys):
                self.hf_sample_data_io.save(
                    explanation_type,
                    # the explanations are saved by summing over the last dimension
                    # this is because the last feature dimension can be extremely large such as 768 for transformers
                    # this takes too much disk space
                    explanation[sample_index].detach().cpu().sum(-1).numpy(),
                    sample_key,
                )

        if verify_outputs:
            loaded_outputs = self.load_outputs(sample_keys)
            for explanation, loaded_explanation in zip(
                outputs[EXPLANATIONS_KEY], loaded_outputs["explanations"]
            ):
                assert (
                    explanation.detach().cpu().sum(-1).cpu() - loaded_explanation
                ).norm() < 1e-10

    def load_outputs(self, sample_keys: List[str]) -> Tuple[torch.Tensor | np.ndarray]:
        outputs = {}
        outputs[EXPLANATIONS_KEY] = tuple(
            [
                torch.cat(
                    [
                        torch.tensor(
                            self.hf_sample_data_io.load(explanation_type, sample_key)
                        )
                        for sample_key in sample_keys
                    ]
                )
                for explanation_type in self.explanation_type_keys
            ]
        )
        outputs[CONVERGENCE_DELTA_KEY] = [
            self.hf_sample_data_io.load(CONVERGENCE_DELTA_KEY, sample_key)
            for sample_key in sample_keys
        ]
        if np.any(outputs[CONVERGENCE_DELTA_KEY] == None):
            outputs.pop(CONVERGENCE_DELTA_KEY)

        return outputs
