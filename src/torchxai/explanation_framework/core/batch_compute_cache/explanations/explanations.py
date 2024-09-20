from __future__ import annotations

import inspect
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torchfusion.core.utilities.logging import get_logger

from torchxai.explanation_framework.core.batch_compute_cache.base import (
    BatchComputeCache,
)
from torchxai.explanation_framework.core.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.explanation_framework.core.utils.general import (
    ExplanationParameters,
    unpack_explanation_parameters,
)
from torchxai.explanation_framework.core.utils.h5io import (
    HFIOMultiOutput,
    HFIOSingleOutput,
)

logger = get_logger()

EXPLANATIONS_KEY = "explanations"
CONVERGENCE_DELTA_KEY = "convergence_delta"


class ExplanationsBatchComputeCache(BatchComputeCache):
    def __init__(
        self,
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
        explanation_tuple_keys: Optional[List[str]] = None,
        explanation_reduce_fn: Callable = lambda x: x,
    ) -> None:
        super().__init__(
            metric_name=EXPLANATIONS_KEY,
            hf_sample_data_io=hf_sample_data_io,
        )
        self._explanation_type_keys = (
            explanation_tuple_keys
            if explanation_tuple_keys is not None
            else ("input_0",)
        )
        self._explanation_reduce_fn = explanation_reduce_fn

    def _prepare_explainer_input(
        self,
        explainer: FusionExplainer,
        explanation_parameters: ExplanationParameters,
        batch_target_labels: Union[torch.Tensor, np.ndarray],
        train_baselines: Union[torch.Tensor, np.ndarray],
    ):
        inputs, baselines, feature_masks, additional_forward_args = (
            unpack_explanation_parameters(explanation_parameters)
        )

        # Pass necessary arguments based on the explanation method's requirements
        kwargs = dict(
            inputs=inputs,
            target=batch_target_labels,
            additional_forward_args=additional_forward_args,
        )

        fn_parameters = inspect.signature(explainer.explain).parameters
        if "feature_masks" in fn_parameters:
            kwargs["feature_masks"] = feature_masks
        if "baselines" in fn_parameters:
            kwargs["baselines"] = baselines
        if "train_baselines" in fn_parameters:
            kwargs["train_baselines"] = tuple(train_baselines.values())
            kwargs["train_baselines"] = tuple(
                x.to(inputs[0].device) for x in kwargs["train_baselines"]
            )
        if "return_convergence_delta" in fn_parameters:
            kwargs["return_convergence_delta"] = True

        for k, v in kwargs.items():
            if isinstance(v, tuple):
                for i, tensor in enumerate(v):
                    logger.debug(f"Key: {k}[{i}], Value: {tensor.shape}")
            elif isinstance(v, torch.Tensor):
                logger.debug(f"Key: {k}, Value: {v.shape}")
            else:
                logger.debug(f"Key: {k}, Value: {v}")

        return kwargs

    def compute_metric(
        self,
        explainer: FusionExplainer,
        explanation_parameters: ExplanationParameters,
        batch_target_labels: Union[torch.Tensor, np.ndarray],
        train_baselines: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[List[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]]:
        # prepare the input for the explainer
        kwargs = self._prepare_explainer_input(
            explainer, explanation_parameters, batch_target_labels, train_baselines
        )

        # generate the explanations
        explanation_outputs = explainer.explain(**kwargs)

        # prepare the output for saving
        if isinstance(explanation_outputs, tuple) and len(explanation_outputs) == 2:
            return {
                EXPLANATIONS_KEY: (explanation_outputs[0]),
                CONVERGENCE_DELTA_KEY: (explanation_outputs[1]),
            }
        else:
            return {
                EXPLANATIONS_KEY: explanation_outputs,
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

        explanations = outputs[EXPLANATIONS_KEY]

        # reduce the explanation if a reduction function is provided
        reduced_explanations = self._explanation_reduce_fn(explanations)

        if not isinstance(reduced_explanations, tuple):
            reduced_explanations = (reduced_explanations,)

        # Save each explanation for each sample key
        for explanation_type, reduced_explanation in zip(
            self._explanation_type_keys, reduced_explanations
        ):
            for sample_index, sample_key in enumerate(sample_keys):
                self.hf_sample_data_io.save(
                    explanation_type,
                    # the explanations are saved by summing over the last dimension
                    # this is because the last feature dimension can be extremely large such as 768 for transformers
                    # this takes too much disk space
                    reduced_explanation[sample_index].detach().cpu().numpy(),
                    sample_key,
                )

        if verify_outputs:
            loaded_outputs = self.load_outputs(sample_keys)
            for reduced_explanation, loaded_explanation in zip(
                reduced_explanations, loaded_outputs["explanations"]
            ):
                assert (
                    reduced_explanation.detach().cpu() - loaded_explanation
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
                for explanation_type in self._explanation_type_keys
            ]
        )
        outputs[CONVERGENCE_DELTA_KEY] = [
            self.hf_sample_data_io.load(CONVERGENCE_DELTA_KEY, sample_key)
            for sample_key in sample_keys
        ]
        if np.any(outputs[CONVERGENCE_DELTA_KEY] == None):
            outputs.pop(CONVERGENCE_DELTA_KEY)

        return outputs
