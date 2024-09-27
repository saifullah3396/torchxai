from pathlib import Path
from typing import Union

from torchxai.explainers.batch_computation_handler.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputationHandler,
)
from torchxai.explainers.utils.constants import ExplanationMetrics


class SparsenessBatchComputationHandler(TorchXAIMetricBatchComputationHandler):
    def __init__(
        self,
        output_file: Union[str, Path],
    ) -> None:
        super().__init__(
            metric_name=ExplanationMetrics.SPARSENESS.value,
            output_file=output_file,
        )

    def _get_metric_fn(self):
        from torchxai.metrics import sparseness

        return sparseness
