from pathlib import Path
from typing import Union

from torchxai.explanation_framework.batch_computation_handler.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputationHandler,
)
from torchxai.explanation_framework.utils.constants import ExplanationMetrics


class CompletenessBatchComputationHandler(TorchXAIMetricBatchComputationHandler):
    def __init__(
        self,
        output_file: Union[str, Path],
    ) -> None:
        super().__init__(
            metric_name=ExplanationMetrics.COMPLETENESS.value,
            output_file=output_file,
        )

    def _get_metric_fn(self):
        from torchxai.metrics import completeness

        return completeness
