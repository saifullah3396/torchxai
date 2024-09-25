from pathlib import Path
from typing import Union

from torchxai.explanation_framework.batch_computation_handler.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputationHandler,
)
from torchxai.explanation_framework.utils.constants import ExplanationMetrics


class MonotonicityBatchComputationHandler(TorchXAIMetricBatchComputationHandler):
    def __init__(
        self,
        output_file: Union[str, Path],
        max_features_processed_per_batch: int = 10,
        show_progress: bool = False,
    ) -> None:
        super().__init__(
            metric_name=ExplanationMetrics.MONOTONICITY.value,
            output_file=output_file,
            output_keys=["monotonicity_scores", "asc_baseline_perturbed_fwds_batch"],
            max_features_processed_per_batch=max_features_processed_per_batch,
            show_progress=show_progress,
        )

    def _get_metric_fn(self):
        from torchxai.metrics import monotonicity

        return monotonicity
