from pathlib import Path
from typing import Any, Mapping, Union

from torchfusion.core.utilities.logging import get_logger

from torchxai.explainers.batch_computation_handler.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputationHandler,
)
from torchxai.explainers.utils.constants import ExplanationMetrics

logger = get_logger()


class AOPCBatchComputationHandler(TorchXAIMetricBatchComputationHandler):
    def __init__(
        self,
        output_file: Union[str, Path],
        max_features_processed_per_batch: int = 10,
        total_features_perturbed: int = 100,
        n_random_perms: int = 10,
        show_progress: bool = False,
    ) -> None:
        super().__init__(
            metric_name=ExplanationMetrics.AOPC.value,
            output_file=output_file,
            output_keys=[
                "desc",
                "asc",
                "rand",
            ],
            max_features_processed_per_batch=max_features_processed_per_batch,
            total_features_perturbed=total_features_perturbed,
            n_random_perms=n_random_perms,
            show_progress=show_progress,
        )

    def _log_outputs(self, outputs: Mapping[str, Any]):
        for key in self._output_keys:
            logger.info(
                f"{key} scores: {[x.mean(-1) for x in outputs[key]]}, shape: {[x.shape for x in outputs[key]]}"
            )

    def _get_metric_fn(self):
        from torchxai.metrics import aopc

        return aopc
