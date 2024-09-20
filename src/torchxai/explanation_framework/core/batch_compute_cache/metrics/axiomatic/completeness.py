from typing import Union

from torchxai.explanation_framework.core.batch_compute_cache.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputeCache,
)
from torchxai.explanation_framework.core.utils.h5io import (
    HFIOMultiOutput,
    HFIOSingleOutput,
)


class CompletenessBatchComputeCache(TorchXAIMetricBatchComputeCache):
    def __init__(
        self,
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
    ) -> None:
        super().__init__(
            metric_name="completeness",
            hf_sample_data_io=hf_sample_data_io,
        )

    def get_metric_fn(self):
        from torchxai.metrics import completeness

        return completeness
