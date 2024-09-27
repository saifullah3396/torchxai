from pathlib import Path
from typing import Union

from torchxai.explainers.batch_computation_handler.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputationHandler,
)
from torchxai.explainers.utils.constants import ExplanationMetrics


class EffectiveComplexityBatchComputationHandler(TorchXAIMetricBatchComputationHandler):
    def __init__(
        self,
        output_file: Union[str, Path],
        eps: float = 1.0e-5,
        normalize_attribution: bool = True,
    ) -> None:
        super().__init__(
            metric_name=ExplanationMetrics.EFFECTIVE_COMPLEXITY.value,
            output_file=output_file,
            eps=eps,
            normalize_attribution=normalize_attribution,
        )

    def _get_metric_fn(self):
        from torchxai.metrics import effective_complexity

        return effective_complexity
