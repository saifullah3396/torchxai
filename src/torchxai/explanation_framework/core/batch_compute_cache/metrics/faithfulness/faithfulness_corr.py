from typing import Any, Callable, List, Tuple, Union

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
from torchxai.metrics._utils.perturbation import default_random_perturb_func


class FaithfulnessCorrelationBatchComputeCache(TorchXAIMetricBatchComputeCache):
    def __init__(
        self,
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
        pertub_func_type: str = "random",
        perturbation_noise_scale: float = 0.02,
        n_perturb_samples: int = 10,
        max_examples_per_batch: int = None,
        perturbation_probability: float = 0.1,
    ) -> None:
        super().__init__(
            metric_name="faithfulness_corr",
            hf_sample_data_io=hf_sample_data_io,
            n_perturb_samples=n_perturb_samples,
            max_examples_per_batch=max_examples_per_batch,
            perturbation_probability=perturbation_probability,
        )

        if pertub_func_type == "random":
            self.perturb_func = default_random_perturb_func(perturbation_noise_scale)
        else:
            raise ValueError(f"Invalid perturbation function type: {pertub_func_type}")

    def get_metric_fn(self):
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
                "faithfulness_corr_scores",
                outputs[self.metric_name][0][sample_index],
                sample_key,
            )

            # save non_sens results
            self.hf_sample_data_io.save(
                "faithfulness_corr_attributions_expanded_perturbed_sum",
                outputs[self.metric_name][1][sample_index],
                sample_key,
            )

            # save the n_features for each input results
            self.hf_sample_data_io.save(
                "faithfulness_corr_perturbed_fwd_diffs",
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
                    "faithfulness_corr_scores",
                    "faithfulness_corr_attributions_expanded_perturbed_sum",
                    "faithfulness_corr_perturbed_fwd_diffs",
                ],
                [0, 1, 2],
            ):
                assert np.allclose(
                    loaded_outputs[output_name],
                    outputs[self.metric_name][output_idx],
                    equal_nan=True,
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
                "faithfulness_corr_scores",
                "faithfulness_corr_attributions_expanded_perturbed_sum",
                "faithfulness_corr_perturbed_fwd_diffs",
            ]
        }
