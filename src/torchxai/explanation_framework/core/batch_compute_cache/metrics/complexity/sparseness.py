from typing import Callable, Tuple, Union

import numpy as np
import torch

from torchxai.explanation_framework.core.batch_compute_cache.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputeCache,
)
from torchxai.explanation_framework.core.utils.general import ExplanationParameters
from torchxai.explanation_framework.core.utils.h5io import (
    HFIOMultiOutput,
    HFIOSingleOutput,
)


class SparsenessBatchComputeCache(TorchXAIMetricBatchComputeCache):
    def __init__(
        self,
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
    ) -> None:
        super().__init__(
            metric_name="sparseness",
            hf_sample_data_io=hf_sample_data_io,
        )

    def _prepare_metric_input(
        self,
        wrapped_model: Callable,
        explanations: Tuple[Union[torch.Tensor, np.ndarray], ...],
        explanation_parameters: ExplanationParameters,
        batch_target_labels: Union[torch.Tensor, np.ndarray],
    ):
        kwargs = {}
        if isinstance(explanations, tuple):
            kwargs["attributions"] = tuple(
                explanation.to(batch_target_labels.device)
                for explanation in explanations
            )
        else:
            kwargs["attributions"] = explanations.to(batch_target_labels.device)
        return kwargs

    def get_metric_fn(self):
        from torchxai.metrics import sparseness

        return sparseness
