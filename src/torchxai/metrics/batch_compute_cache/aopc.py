from __future__ import annotations

import inspect
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from captum.metrics import aopc_max
from torchfusion.core.utilities.logging import get_logger

from doclm.explanation_framework.core.batch_compute_cache.base import BatchComputeCache
from doclm.explanation_framework.core.captum_attribution_handlers.base_handler import (
    CaptumAttributionBaseHandler,
)
from doclm.explanation_framework.core.utils.h5io import (
    HFIOMultiOutput,
    HFIOSingleOutput,
)

logger = get_logger()

AOPC_KEY = "aopc"


class AOPCBatchComputeCache(BatchComputeCache):
    def __init__(
        self,
        metric_name: str,
        group_key: str,
        output_keys: List[str],
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
        perturb_radius: float = 0.02,
        n_perturb_samples: int = 50,
    ) -> None:
        super().__init__(
            metric_name=metric_name,
            group_key=group_key,
            output_keys=output_keys,
            hf_sample_data_io=hf_sample_data_io,
        )
        self.perturb_radius = perturb_radius
        self.n_perturb_samples = n_perturb_samples

    def compute_metric(
        self,
        explanation_method: CaptumAttributionBaseHandler,
        model_inputs: Tuple[Union[torch.Tensor, np.ndarray], ...],
        batch_target_labels: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[List[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]]:
        (
            inputs,
            baselines,
            feature_masks,
            attention_mask,
            token_type_embeddings,
            bbox,
        ) = model_inputs

        # make sure that the batch size is the same for all inputs and target labels
        assert (
            inputs[0].shape[0] == batch_target_labels.shape[0]
            and inputs[0].shape[0] == baselines[0].shape[0]
        ), "Input, baselines and target labels shape must match"

        # Pass necessary arguments based on the attribution method's requirements
        args = dict(
            inputs=inputs,
            target=batch_target_labels,
            additional_forward_args=(attention_mask, token_type_embeddings, bbox),
        )

        fn_parameters = inspect.signature(explanation_method.attribute).parameters
        if "feature_masks" in fn_parameters:
            args["feature_masks"] = feature_masks
        if "baselines" in fn_parameters:
            args["baselines"] = baselines
        if "return_convergence_delta" in fn_parameters:
            args["return_convergence_delta"] = False

        # compute metrics
        aopc_scores = (
            aopc_max(
                explanation_method.attribute,
                **args,
                perturb_radius=self.perturb_radius,  # perturb each input by 0.02
                n_perturb_samples=self.n_perturb_samples,  # compute aopc for only 50 perturbations because it is too expensive otherwise
                max_examples_per_batch=batch_target_labels.shape[0],
            )
            .detach()
            .cpu()
            .numpy()
        )

        return {AOPC_KEY: aopc_scores}

    def save_outputs(
        self,
        sample_keys: List[str],
        outputs: dict[Any],
    ) -> None:
        # save aopc scores in output file
        for sample_index, sample_key in enumerate(sample_keys):
            # save aopc results
            self.hf_sample_data_io.save(
                AOPC_KEY,
                outputs[AOPC_KEY][sample_index],
                sample_key,
            )

            # save aopc setup attributes
            self.hf_sample_data_io.save_attribute(
                "perturb_radius", self.perturb_radius, sample_key
            )
            self.hf_sample_data_io.save_attribute(
                "n_perturb_samples", self.n_perturb_samples, sample_key
            )
