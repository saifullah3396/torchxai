from pathlib import Path
from typing import Any, Mapping, Union

from torchfusion.core.utilities.logging import get_logger

from torchxai.explanation_framework.batch_computation_handler.metrics.torchxai_metrics import (
    TorchXAIMetricBatchComputationHandler,
)
from torchxai.explanation_framework.utils.constants import ExplanationMetrics

logger = get_logger()


class MonotonicityCorrNonSensitivityBatchComputationHandler(
    TorchXAIMetricBatchComputationHandler
):
    def __init__(
        self,
        output_file: Union[str, Path],
        n_perturbations_per_feature: int = 1,
        max_features_processed_per_example: int = 10,
        eps: float = 0.00001,
        show_progress: bool = False,
    ) -> None:
        super().__init__(
            metric_name=ExplanationMetrics.MONOTONICITY_CORR_AND_NON_SENS.value,
            output_file=output_file,
            output_keys=[
                "monotonicity_correlation",
                "non_sensitivity",
                "n_features",
            ],
            max_features_processed_per_example=max_features_processed_per_example,
            n_perturbations_per_feature=n_perturbations_per_feature,
            eps=eps,
            show_progress=show_progress,
        )

    def _log_outputs(self, outputs: Mapping[str, Any]):
        for key in self._output_keys[:2]:
            logger.info(f"{key} scores: {outputs[key]}")

    def _get_metric_fn(self):
        from torchxai.metrics import monotonicity_corr_and_non_sens

        return monotonicity_corr_and_non_sens
