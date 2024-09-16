from __future__ import annotations

from typing import Any, Callable, List, Tuple, Union

import numpy as np
import quantus
import torch
from torchfusion.core.utilities.logging import get_logger

from doclm.explanation_framework.core.batch_compute_cache.base import BatchComputeCache
from doclm.explanation_framework.core.utils.general import unpack_inputs
from doclm.explanation_framework.core.utils.h5io import (
    HFIOMultiOutput,
    HFIOSingleOutput,
)

logger = get_logger()
def completeness()

                infidelity(
                    wrapped_model,
                    perturb_fn_drop,
                    inputs=inputs,
                    attributions=explanations,
                    baselines=baselines,
                    additional_forward_args=(
                        # this tuple in this order must be passed see src/doclm/models/interpretable_layoutlmv3.py
                        # for forward method
                        extra_inputs[EMBEDDING_KEYS.TOKEN_TYPE_EMBEDDINGS],
                        extra_inputs[DataKeys.ATTENTION_MASKS],
                        extra_inputs[DataKeys.TOKEN_BBOXES],
                    ),
                    target=batch_target_labels,
                    max_examples_per_batch=batch_target_labels.shape[0],
                    n_perturb_samples=self.n_perturb_samples,
                    normalize=True,
                )
class Completeness(object):
    def __call__(
        self,
        model: Callable,
        x_batch: Tuple[torch.Tensor, ...],
        y_batch: Union[torch.Tensor, np.ndarray],
        a_batch: Tuple[torch.Tensor, ...],
        device: torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        x_input = model.shape_input(x, x.shape, channel_first=True)
        y_pred = float(model.predict(x_input)[:, y])

        # Predict on baseline.
        x_input = model.shape_input(x_baseline, x.shape, channel_first=True)
        y_pred_baseline = float(model.predict(x_input)[:, y])

        if np.sum(a) == self.output_func(y_pred - y_pred_baseline):
            return True
        else:
            return False


class TorchFusionExplMetricFactory:
    @staticmethod
    def create(metric_name: str, metric_kwargs: dict[Any]) -> Metric:
        if metric_name == "completeness":
            default_kwargs = {
                "abs": False,
                "normalise": False,
            }
            kwargs = {**default_kwargs, **metric_kwargs}
            return Completeness(*kwargs)
        else:
            raise ValueError(f"Quantus metric {metric_name} is not supported.")


class TorchFusionExplMetricBatchComputeCache(BatchComputeCache):
    def __init__(
        self,
        metric_name: str,
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
        metric_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            metric_name=f"torchfusion_{metric_name}",
            hf_sample_data_io=hf_sample_data_io,
        )

        # initialize the quantus class
        self.torchfusion_metric = TorchFusionExplMetricFactory.create(
            metric_name.replace("torchfusion_", ""), metric_kwargs
        )

    def compute_metric(
        self,
        wrapped_model: Callable,
        explanations: Tuple[Union[torch.Tensor, np.ndarray], ...],
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
        ), "Input, baselines, and target labels shape must match"

        metric_scores = self.torchfusion_metric(
            model=wrapped_model,
            x_batch=inputs,
            y_batch=batch_target_labels,
            a_batch=explanations,
            device=inputs[0].device,
        )
        return metric_scores

    def save_outputs(
        self,
        sample_keys: List[str],
        outputs: dict[Any],
        time_taken: float,
        verify_outputs: bool = True,
    ) -> None:
        super().save_outputs(sample_keys, outputs, time_taken)

        for dropout_probability in self.drop_probabilities:
            # save infidelity scores in output file
            for sample_index, sample_key in enumerate(sample_keys):
                # save infidelity results
                self.hf_sample_data_io.save(
                    f"{INFIDELITY_KEY}_{dropout_probability}",
                    outputs[f"{INFIDELITY_KEY}_{dropout_probability}"][sample_index],
                    sample_key,
                )

                # save infidelity setup attributes
                self.hf_sample_data_io.save_attribute(
                    "perturb_noise_scale", self.perturb_noise_scale, sample_key
                )
                self.hf_sample_data_io.save_attribute(
                    "n_perturb_samples", self.n_perturb_samples, sample_key
                )

        if verify_outputs:
            loaded_outputs = self.load_outputs(sample_keys)
            print(loaded_outputs, outputs)
            exit()

    def load_outputs(self, sample_keys: List[str]) -> Tuple[torch.Tensor | np.ndarray]:
        return {
            f"{INFIDELITY_KEY}_{dropout_probability}": [
                self.hf_sample_data_io.load(
                    f"{INFIDELITY_KEY}_{dropout_probability}", sample_key
                )
                for sample_key in sample_keys
            ]
            for dropout_probability in self.drop_probabilities
        }
