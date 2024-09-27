from __future__ import annotations

import inspect
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from ignite.utils import convert_tensor
from torchfusion.core.utilities.logging import get_logger

from torchxai.explainers.batch_computation_handler.base import BatchComputationHandler
from torchxai.explainers.utils.common import (
    ExplanationParameters,
    _expand_feature_mask_to_target,
    unpack_explanation_parameters,
)

logger = get_logger()


class TorchXAIMetricBatchComputationHandler(BatchComputationHandler):
    def __init__(
        self,
        metric_name: str,
        output_file: Union[str, Path],
        output_keys: Optional[List[str]] = None,
        **metric_kwargs,
    ) -> None:
        super().__init__(
            metric_name=metric_name,
            output_file=output_file,
        )

        # save the metric kwargs
        self._metric_kwargs = metric_kwargs

        # initialize the quantus class
        self._torchxai_metric = self._get_metric_fn()

        # initialize the default output_transform
        self._output_keys = output_keys
        if self._output_keys is None:
            self._output_keys = [self._metric_name]

        def _output_transform(output):
            if not isinstance(output, tuple):
                output = (output,)
            return {k: v for k, v in zip(self._output_keys, output)}

        self._output_transform = _output_transform

    @abstractmethod
    def _get_metric_fn(self):
        pass

    def _prepare_metric_input(
        self,
        wrapped_model: Callable,
        explanations: Tuple[Union[torch.Tensor, np.ndarray], ...],
        explanation_parameters: ExplanationParameters,
        batch_target_labels: Union[torch.Tensor, np.ndarray],
    ):
        (
            inputs,
            baselines,
            feature_mask,
            additional_forward_args,
        ) = unpack_explanation_parameters(explanation_parameters)

        assert (
            inputs[0].shape[0] == batch_target_labels.shape[0],
            f"Input and target labels shape must match, target labels shape: {batch_target_labels.shape}, input shape: {inputs[0].shape}",
        )
        assert (
            inputs[0].shape[0] == baselines[0].shape[0]
        ), f"Input and baselines shape must match, baselines shape: {baselines[0].shape}, input shape: {inputs[0].shape}"
        assert (
            inputs[0].shape[0] == explanations[0].shape[0]
        ), f"Input and explanations shape must match, explanations shape: {explanations[0].shape}, input shape: {inputs[0].shape}"

        # Pass necessary arguments based on the explanation method's requirements
        kwargs = dict()
        device = batch_target_labels.device
        fn_parameters = inspect.signature(self._torchxai_metric).parameters
        if "inputs" in fn_parameters:
            kwargs["inputs"] = inputs
        if "target" in fn_parameters:
            kwargs["target"] = batch_target_labels
        if "additional_forward_args" in fn_parameters:
            kwargs["additional_forward_args"] = additional_forward_args
        if "forward_func" in fn_parameters:
            kwargs["forward_func"] = wrapped_model
        if "attributions" in fn_parameters:
            kwargs["attributions"] = convert_tensor(explanations, device=device)
        if "feature_mask" in fn_parameters:
            feature_mask = _expand_feature_mask_to_target(feature_mask, explanations)
            kwargs["feature_mask"] = convert_tensor(feature_mask, device=device)
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

    def _compute_metric(
        self,
        wrapped_model: Callable,
        explanations: Tuple[Union[torch.Tensor, np.ndarray], ...],
        explanation_parameters: ExplanationParameters,
        batch_target_labels: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[List[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]]:
        # prepare inputs
        kwargs = self._prepare_metric_input(
            wrapped_model, explanations, explanation_parameters, batch_target_labels
        )

        # compute metric
        output = self._torchxai_metric(**kwargs, **self._metric_kwargs)

        # convert to cpu
        output = convert_tensor(output, device="cpu")

        return self._output_transform(output)

    def _log_outputs(self, outputs: Mapping[str, Any]):
        for key in self._output_keys[:1]:
            logger.info(f"{key} scores: {outputs[key]}")

    def _save_sample_output(self, sample_key: str, output_key: str, output: Any):
        self._hfio.save(
            output_key,
            output,
            sample_key,
        )

    def _save_sample_attributes(self, sample_key: str):
        for k, v in self._metric_kwargs.items():
            try:
                self._hfio.save_attribute(k, v, sample_key)
            except Exception as exc:
                logger.exception(f"Exception raised while saving attribute {k}: {exc}")

    def _save_outputs(
        self,
        sample_keys: List[str],
        outputs: dict[Any],
        time_taken: float,
        verify_outputs: bool = True,
    ) -> None:
        super()._save_outputs(sample_keys, outputs, time_taken)
        for sample_index, sample_key in enumerate(sample_keys):
            for k, v in outputs.items():
                self._save_sample_output(
                    sample_key, output_key=k, output=v[sample_index]
                )
            self._save_sample_attributes(sample_key)

        if (
            verify_outputs
        ):  # this is only for sanity check, may be removed in production
            loaded_outputs = self._load_outputs(sample_keys)
            for output_key in self._output_keys:
                for loaded, original in zip(
                    loaded_outputs[output_key], outputs[output_key]
                ):
                    assert np.allclose(
                        loaded, original, equal_nan=True
                    ), f"Loaded outputs do not match saved outputs: {loaded} != {original}"

    def _load_outputs(self, sample_keys: List[str]) -> Tuple[torch.Tensor | np.ndarray]:
        return {
            output_key: [
                self._hfio.load(output_key, sample_key) for sample_key in sample_keys
            ]
            for output_key in self._output_keys
        }
