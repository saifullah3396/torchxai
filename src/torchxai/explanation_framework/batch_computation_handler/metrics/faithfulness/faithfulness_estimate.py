from pathlib import Path
from typing import Union

from torchxai.explainers.batch_computation_handler.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputationHandler,
)
from torchxai.explainers.utils.constants import ExplanationMetrics


class FaithfulnessEstimateBatchComputationHandler(
    TorchXAIMetricBatchComputationHandler
):
    def __init__(
        self,
        output_file: Union[str, Path],
        max_features_processed_per_batch: int = 10,
        show_progress: bool = False,
    ) -> None:
        super().__init__(
            metric_name=ExplanationMetrics.FAITHFULNESS_ESTIMATE.value,
            output_file=output_file,
            max_features_processed_per_batch=max_features_processed_per_batch,
            show_progress=show_progress,
            output_keys=[
                "faithfulness_estimate",
                "attributions_sum_perturbed",
                "inputs_perturbed_fwd_diffs",
            ],
        )

    def _get_metric_fn(self):
        from torchxai.metrics import faithfulness_estimate

        return faithfulness_estimate
