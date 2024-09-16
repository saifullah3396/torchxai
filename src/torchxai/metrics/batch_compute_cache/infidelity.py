from __future__ import annotations

from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
import tqdm
from captum.metrics import infidelity
from torchfusion.core.constants import DataKeys
from torchfusion.core.utilities.logging import get_logger

from doclm.explanation_framework.core.batch_compute_cache.base import BatchComputeCache
from doclm.explanation_framework.core.utils.constants import EMBEDDING_KEYS
from doclm.explanation_framework.core.utils.general import (
    get_feature_groups_and_counts,
    perturb_fn_drop_batched_single_output,
    unpack_inputs,
)
from doclm.explanation_framework.core.utils.h5io import (
    HFIOMultiOutput,
    HFIOSingleOutput,
)

logger = get_logger()

INFIDELITY_KEY = "infidelity"


class InfidelityBatchComputeCache(BatchComputeCache):
    def __init__(
        self,
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
        drop_probabilities: List[float] = [0.1, 0.25, 0.5],
        perturb_noise_scale: float = 0.02,
        n_perturb_samples: int = 10,
    ) -> None:
        super().__init__(
            metric_name=INFIDELITY_KEY,
            hf_sample_data_io=hf_sample_data_io,
        )
        self.drop_probabilities = drop_probabilities
        self.perturb_noise_scale = perturb_noise_scale
        self.n_perturb_samples = n_perturb_samples

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

        # compute infidelity perturbation function which takes cares of grouped features
        grouped_feature_counts, n_grouped_features = get_feature_groups_and_counts(
            feature_masks
        )

        infidelity_scores_dict = {}
        logger.debug("Computing infidelity scores for different drop probabilities")
        for drop_probability in tqdm.tqdm(
            self.drop_probabilities, desc="Drop Probabilities"
        ):
            perturb_fn_drop = perturb_fn_drop_batched_single_output(
                grouped_feature_counts,
                n_grouped_features,
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
