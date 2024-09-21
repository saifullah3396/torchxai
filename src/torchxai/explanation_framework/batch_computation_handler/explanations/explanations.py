from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

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
from torchxai.explanation_framework.utils.common import unpack_explanation_parameters
from torchxai.explanation_framework.utils.containers import ExplanationParameters

logger = get_logger()

EXPLANATIONS_KEY = "explanations"
CONVERGENCE_DELTA_KEY = "convergence_delta"


class ExplanationsBatchComputationHandler(BatchComputationHandler):
    def __init__(
        self,
        output_file: Union[str, Path],
        explanation_tuple_keys: Optional[List[str]] = None,
        explanation_reduce_fn: Callable = lambda x: x,
        return_convergence_delta: bool = False,
    ) -> None:
        super().__init__(
            metric_name=EXPLANATIONS_KEY,
            output_file=output_file,
        )
        self._explanation_type_keys = (
            explanation_tuple_keys
            if explanation_tuple_keys is not None
            else ("input_0",)
        )
        self._explanation_reduce_fn = explanation_reduce_fn
        self._return_convergence_delta = return_convergence_delta

        # initialize the default output_transform
        self._output_keys = (
            [EXPLANATIONS_KEY, CONVERGENCE_DELTA_KEY]
            if self._return_convergence_delta
            else [EXPLANATIONS_KEY]
        )

        def _output_transform(output):
            if not self._return_convergence_delta:
                output = (output,)

            return {k: v for k, v in zip(self._output_keys, output)}

        self._output_transform = _output_transform

    def _prepare_explainer_input(
        self,
        explainer: FusionExplainer,
        explanation_parameters: ExplanationParameters,
        batch_target_labels: Union[torch.Tensor, np.ndarray],
        train_baselines: Union[torch.Tensor, np.ndarray],
    ):
        inputs, baselines, feature_mask, additional_forward_args = (
            unpack_explanation_parameters(explanation_parameters)
        )

        # Pass necessary arguments based on the explanation method's requirements
        kwargs = dict(
            inputs=inputs,
            target=batch_target_labels,
            additional_forward_args=additional_forward_args,
        )

        fn_parameters = inspect.signature(explainer.explain).parameters
        device = batch_target_labels.device
        if "feature_mask" in fn_parameters:
            kwargs["feature_mask"] = convert_tensor(feature_mask, device=device)
        if "baselines" in fn_parameters:
            kwargs["baselines"] = baselines
        if "train_baselines" in fn_parameters:
            kwargs["train_baselines"] = convert_tensor(
                tuple(train_baselines.values()), device=device
            )
        if "return_convergence_delta" in fn_parameters:
            kwargs["return_convergence_delta"] = self._return_convergence_delta

        for k, v in kwargs.items():
            if isinstance(v, tuple):
                for i, tensor in enumerate(v):
                    logger.debug(f"Key: {k}[{i}], Value: {tensor.shape}")
            elif isinstance(v, torch.Tensor):
                logger.debug(f"Key: {k}, Value: {v.shape}")
            else:
                logger.debug(f"Key: {k}, Value: {v}")

        return kwargs

    def _compute_metric(
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

        # perform output transform
        return self._output_transform(explanation_outputs)

    def _log_outputs(self, outputs: Mapping[str, Any]):
        explanations = outputs[EXPLANATIONS_KEY]
        logger.info(
            f"Computed explanation output of shape: {tuple(x.shape for x in explanations)}"
        )

    def _save_convergence_deltas(
        self, sample_keys: List[str], convergence_delta: torch.Tensor
    ):
        # Save convergence delta if it's available
        convergence_delta = convergence_delta.cpu().numpy()
        for sample_index, sample_key in enumerate(sample_keys):
            self._hfio.save(
                CONVERGENCE_DELTA_KEY,
                convergence_delta[sample_index],
                sample_key,
            )

    def _save_explanations(self, sample_keys: List[str], explanations: torch.Tensor):
        # reduce the explanation if a reduction function is provided
        reduced_explanations = self._explanation_reduce_fn(explanations)

        if not isinstance(reduced_explanations, tuple):
            reduced_explanations = (reduced_explanations,)

        # Save each explanation for each sample key
        for explanation_type, reduced_explanation in zip(
            self._explanation_type_keys, reduced_explanations
        ):
            for sample_index, sample_key in enumerate(sample_keys):
                self._hfio.save(
                    explanation_type,
                    # the explanations are saved by summing over the last dimension
                    # this is because the last feature dimension can be extremely large such as 768 for transformers
                    # this takes too much disk space
                    reduced_explanation[sample_index].detach().cpu().numpy(),
                    sample_key,
                )

    def _save_outputs(
        self,
        sample_keys: List[str],
        outputs: Mapping[str, torch.Tensor],
        time_taken: float,
        verify_outputs: bool = False,
    ) -> None:
        super()._save_outputs(sample_keys, outputs, time_taken)
        if CONVERGENCE_DELTA_KEY in self._output_keys:
            self._save_convergence_deltas(sample_keys, outputs[CONVERGENCE_DELTA_KEY])
        self._save_explanations(sample_keys, outputs[EXPLANATIONS_KEY])

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
        outputs = {}
        outputs[EXPLANATIONS_KEY] = tuple(
            [
                torch.cat(
                    [
                        torch.tensor(self._hfio.load(explanation_type, sample_key))
                        for sample_key in sample_keys
                    ]
                )
                for explanation_type in self._explanation_type_keys
            ]
        )
        if CONVERGENCE_DELTA_KEY in self._output_keys:
            outputs[CONVERGENCE_DELTA_KEY] = [
                self._hfio.load(CONVERGENCE_DELTA_KEY, sample_key)
                for sample_key in sample_keys
            ]
        return outputs
