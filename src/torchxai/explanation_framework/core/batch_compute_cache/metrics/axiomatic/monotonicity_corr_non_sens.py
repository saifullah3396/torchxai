from typing import Any, List, Tuple, Union

import numpy as np
import torch

from torchxai.explanation_framework.core.batch_compute_cache.base import (
    BatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputeCache,
)
from torchxai.explanation_framework.core.utils.h5io import (
    HFIOMultiOutput,
    HFIOSingleOutput,
)


class MonotonicityCorrNonSensitivityBatchComputeCache(TorchXAIMetricBatchComputeCache):
    def __init__(
        self,
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
        max_features_processed_per_example: int = 10,
        n_perturbations_per_feature: int = 1,
    ) -> None:
        super().__init__(
            metric_name="monotonicity_corr_and_non_sens",
            hf_sample_data_io=hf_sample_data_io,
            max_features_processed_per_example=max_features_processed_per_example,
            n_perturbations_per_feature=n_perturbations_per_feature,
        )

    def get_metric_fn(self):
        from torchxai.metrics import monotonicity_corr_and_non_sens

        return monotonicity_corr_and_non_sens

    def save_outputs(
        self,
        sample_keys: List[str],
        outputs: dict[Any],
        time_taken: float,
        verify_outputs: bool = True,
    ) -> None:
        BatchComputeCache.save_outputs(self, sample_keys, outputs, time_taken)

        # save infidelity scores in output file
        for sample_index, sample_key in enumerate(sample_keys):
            # save monotonicity_corr results
            self.hf_sample_data_io.save(
                "torchxai_monotonicity_correlation",
                outputs[self.metric_name][0][sample_index],
                sample_key,
            )

            # save non_sens results
            self.hf_sample_data_io.save(
                "torchxai_non_sensitivity",
                outputs[self.metric_name][1][sample_index],
                sample_key,
            )

            # save the n_features for each input results
            self.hf_sample_data_io.save(
                "torchxai_non_sensitivity_n_features",
                outputs[self.metric_name][2][sample_index],
                sample_key,
            )

            # save setup attributes
            for k, v in self.metric_kwargs.items():
                self.hf_sample_data_io.save_attribute(k, v, sample_key)

        if (
            verify_outputs
        ):  # this is only for sanity check, may be removed in production
            loaded_outputs = self.load_outputs(sample_keys)
            for output_name, output_idx in zip(
                [
                    "torchxai_monotonicity_correlation",
                    "torchxai_non_sensitivity",
                    "torchxai_non_sensitivity_n_features",
                ],
                [0, 1, 2],
            ):
                assert np.allclose(
                    loaded_outputs[output_name], outputs[self.metric_name][output_idx]
                ), f"Loaded outputs do not match saved outputs: {loaded_outputs[output_name]} != {outputs[self.metric_name][output_idx]}"

    def load_outputs(self, sample_keys: List[str]) -> Tuple[torch.Tensor | np.ndarray]:
        return {
            k: np.squeeze(
                np.array(
                    [
                        self.hf_sample_data_io.load(k, sample_key)
                        for sample_key in sample_keys
                    ]
                )
            )
            for k in [
                "torchxai_monotonicity_correlation",
                "torchxai_non_sensitivity",
                "torchxai_non_sensitivity_n_features",
            ]
        }
