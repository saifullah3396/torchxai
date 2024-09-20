from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
from torchfusion.core.constants import DataKeys
from torchfusion.core.utilities.logging import get_logger

from torchxai.explanation_framework.core.batch_compute_cache.base import (
    BatchComputeCache,
)
from torchxai.explanation_framework.core.utils.constants import EMBEDDING_KEYS
from torchxai.explanation_framework.core.utils.general import unpack_inputs
from torchxai.explanation_framework.core.utils.h5io import (
    HFIOMultiOutput,
    HFIOSingleOutput,
)

logger = get_logger()


class TorchXAIMetricBatchComputeCache(BatchComputeCache):
    def __init__(
        self,
        metric_name: str,
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
        **metric_kwargs,
    ) -> None:
        super().__init__(
            metric_name=f"torchxai_{metric_name}",
            hf_sample_data_io=hf_sample_data_io,
        )

        # save the metric kwargs
        self.metric_kwargs = metric_kwargs

        # initialize the quantus class
        self.torchxai_metric = self.get_metric_fn()

    @abstractmethod
    def get_metric_fn(self):
        pass

    def _prepare_metric_input(
        self,
        wrapped_model: Callable,
        explanations: Tuple[Union[torch.Tensor, np.ndarray], ...],
        model_inputs: Tuple[Union[torch.Tensor, np.ndarray], ...],
        batch_target_labels: Union[torch.Tensor, np.ndarray],
    ):
        (
            inputs,
            baselines,
            feature_masks,
            extra_inputs,
            _,
        ) = unpack_inputs(model_inputs)

        # Pass necessary arguments based on the explanation method's requirements
        kwargs = dict(
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

        if isinstance(feature_masks, tuple):
            feature_masks = tuple(
                (
                    mask.unsqueeze(-1)
                    if len(mask.shape) != len(explanation.shape)
                    else mask
                )
                for mask, explanation in zip(feature_masks, explanations)
            )
            feature_masks = tuple(
                mask.expand_as(explanation)
                for mask, explanation in zip(feature_masks, explanations)
            )
        else:
            feature_masks = (
                feature_masks.unsqueeze(-1)
                if len(feature_masks.shape) != len(explanations.shape)
                else feature_masks
            )
        fn_parameters = inspect.signature(self.torchxai_metric).parameters
        if "forward_func" in fn_parameters:
            kwargs["forward_func"] = wrapped_model
        if "attributions" in fn_parameters:
            if isinstance(explanations, tuple):
                kwargs["attributions"] = tuple(
                    explanation.to(input.device)
                    for explanation, input in zip(explanations, inputs)
                )
            else:
                kwargs["attributions"] = explanations.to(inputs.device)
        if "feature_masks" in fn_parameters:
            kwargs["feature_masks"] = feature_masks
        if "baselines" in fn_parameters:
            kwargs["baselines"] = baselines

        for k, v in kwargs.items():
            if isinstance(v, tuple):
                for i, tensor in enumerate(v):
                    logger.debug(
                        f"Key: {k}[{i}], Value: {tensor.shape}, Device: {tensor.device}"
                    )
            elif isinstance(v, torch.Tensor):
                logger.debug(f"Key: {k}, Value: {v.shape}, Device: {v.device}")
            else:
                logger.debug(f"Key: {k}, Value: {v}")

        assert (
            inputs[0].shape[0] == batch_target_labels.shape[0]
            and inputs[0].shape[0] == baselines[0].shape[0]
        ), "Input, baselines, and target labels shape must match"

        return kwargs

    def compute_metric(
        self,
        wrapped_model: Callable,
        explanations: Tuple[Union[torch.Tensor, np.ndarray], ...],
        model_inputs: Tuple[Union[torch.Tensor, np.ndarray], ...],
        batch_target_labels: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[List[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]]:
        kwargs = self._prepare_metric_input(
            wrapped_model, explanations, model_inputs, batch_target_labels
        )

        metric_scores = self.torchxai_metric(**kwargs, **self.metric_kwargs)

        if isinstance(metric_scores, tuple):
            return {
                self.metric_name: tuple(
                    x.cpu().numpy() if isinstance(x, torch.Tensor) else x
                    for x in metric_scores
                )
            }
        else:
            return {self.metric_name: metric_scores.cpu().numpy()}

    def save_outputs(
        self,
        sample_keys: List[str],
        outputs: dict[Any],
        time_taken: float,
        verify_outputs: bool = True,
    ) -> None:
        super().save_outputs(sample_keys, outputs, time_taken)

        # save infidelity scores in output file
        for sample_index, sample_key in enumerate(sample_keys):
            # save infidelity results
            self.hf_sample_data_io.save(
                self.metric_name,
                outputs[self.metric_name][sample_index],
                sample_key,
            )

        if (
            verify_outputs
        ):  # this is only for sanity check, may be removed in production
            loaded_outputs = self.load_outputs(sample_keys)
            for k, v in loaded_outputs.items():
                assert np.allclose(
                    v, outputs[k]
                ), f"Loaded outputs do not match saved outputs: {v} != {outputs[k]}"

    def load_outputs(self, sample_keys: List[str]) -> Tuple[torch.Tensor | np.ndarray]:
        return {
            self.metric_name: [
                self.hf_sample_data_io.load(self.metric_name, sample_key)
                for sample_key in sample_keys
            ]
        }
