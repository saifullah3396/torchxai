from __future__ import annotations

from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
import tqdm
from captum.metrics import infidelity
from torchfusion.core.utilities.logging import get_logger

from torchxai.explanation_framework.core.batch_compute_cache.base import (
    BatchComputeCache,
)
from torchxai.explanation_framework.core.utils.general import (
    ExplanationParameters,
    unpack_explanation_parameters,
)
from torchxai.explanation_framework.core.utils.h5io import (
    HFIOMultiOutput,
    HFIOSingleOutput,
)
from torchxai.metrics._utils.perturbation import perturb_fn_drop_batched_single_output

logger = get_logger()

INFIDELITY_KEY = "infidelity"


class InfidelityBatchComputeCache(BatchComputeCache):
    def __init__(
        self,
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
        drop_probabilities: List[float] = [0.1, 0.25, 0.5],
        perturb_noise_scale: float = 0.02,
        n_perturb_samples: int = 10,
        max_examples_per_batch: int = 100,
    ) -> None:
        super().__init__(
            metric_name=INFIDELITY_KEY,
            hf_sample_data_io=hf_sample_data_io,
        )
        self.drop_probabilities = drop_probabilities
        self.perturb_noise_scale = perturb_noise_scale
        self.n_perturb_samples = n_perturb_samples
        self.max_examples_per_batch = max_examples_per_batch

    def compute_metric(
        self,
        wrapped_model: Callable,
        explanations: Tuple[Union[torch.Tensor, np.ndarray], ...],
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
        ), "Input, baselines, and target labels shape must match"

        # compute infidelity perturbation function which takes cares of grouped features
        if isinstance(explanations, tuple):
            explanations = tuple(
                explanation.to(batch_target_labels.device)
                for explanation in explanations
            )
        else:
            explanations = explanations.to(batch_target_labels.device)

        infidelity_scores_dict = {}
        logger.debug("Computing infidelity scores for different drop probabilities")
        for drop_probability in tqdm.tqdm(
            self.drop_probabilities, desc="Drop Probabilities"
        ):
            perturb_fn_drop = perturb_fn_drop_batched_single_output(
                feature_mask=feature_mask,
                drop_probability=drop_probability,
            )

            # compute infidelity
            infidelity_scores = (
                infidelity(
                    wrapped_model,
                    perturb_fn_drop,
                    inputs=inputs,
                    attributions=explanations,
                    baselines=baselines,
                    additional_forward_args=additional_forward_args,
                    target=batch_target_labels,
                    max_examples_per_batch=self.max_examples_per_batch,
                    n_perturb_samples=self.n_perturb_samples,
                    normalize=True,
                )
                .detach()
                .cpu()
                .numpy()
            )
            infidelity_scores_dict[f"{INFIDELITY_KEY}_{drop_probability}"] = (
                infidelity_scores
            )
        return infidelity_scores_dict

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
            f"{INFIDELITY_KEY}_{dropout_probability}": [
                self.hf_sample_data_io.load(
                    f"{INFIDELITY_KEY}_{dropout_probability}", sample_key
                )
                for sample_key in sample_keys
            ]
            for dropout_probability in self.drop_probabilities
        }
