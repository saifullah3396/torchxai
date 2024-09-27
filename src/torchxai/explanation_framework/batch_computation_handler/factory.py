from __future__ import annotations

from pathlib import Path
from typing import Type

from sympy import Union

from torchxai.explainers.batch_computation_handler.explanations.explanations import (
    ExplanationsBatchComputationHandler,
)
from torchxai.explainers.batch_computation_handler.metrics.axiomatic.completeness import (
    CompletenessBatchComputationHandler,
)
from torchxai.explainers.batch_computation_handler.metrics.axiomatic.monotonicity_corr_non_sens import (
    MonotonicityCorrNonSensitivityBatchComputationHandler,
)
from torchxai.explainers.batch_computation_handler.metrics.complexity.complexity import (
    ComplexityBatchComputationHandler,
)
from torchxai.explainers.batch_computation_handler.metrics.complexity.effective_complexity import (
    EffectiveComplexityBatchComputationHandler,
)
from torchxai.explainers.batch_computation_handler.metrics.complexity.sparseness import (
    SparsenessBatchComputationHandler,
)
from torchxai.explainers.batch_computation_handler.metrics.faithfulness.aopc import (
    AOPCBatchComputationHandler,
)
from torchxai.explainers.batch_computation_handler.metrics.faithfulness.faithfulness_corr import (
    FaithfulnessCorrelationBatchComputationHandler,
)
from torchxai.explainers.batch_computation_handler.metrics.faithfulness.faithfulness_estimate import (
    FaithfulnessEstimateBatchComputationHandler,
)
from torchxai.explainers.batch_computation_handler.metrics.faithfulness.infidelity import (
    InfidelityBatchComputationHandler,
)
from torchxai.explainers.batch_computation_handler.metrics.faithfulness.monotonicity import (
    MonotonicityBatchComputationHandler,
)
from torchxai.explainers.batch_computation_handler.metrics.robustness.sensitivity import (
    SensitivityBatchComputationHandler,
)
from torchxai.explainers.explainer import Explainer
from torchxai.explainers.utils.constants import ExplanationMetrics

AVAILABLE_METRIC_HANDLERS = {
    # axioamtic metrics
    ExplanationMetrics.COMPLETENESS: CompletenessBatchComputationHandler,
    ExplanationMetrics.MONOTONICITY_CORR_AND_NON_SENS: MonotonicityCorrNonSensitivityBatchComputationHandler,
    # complexity metrics
    ExplanationMetrics.COMPLEXITY: ComplexityBatchComputationHandler,
    ExplanationMetrics.EFFECTIVE_COMPLEXITY: EffectiveComplexityBatchComputationHandler,
    ExplanationMetrics.SPARSENESS: SparsenessBatchComputationHandler,
    # faithfulness metrics
    ExplanationMetrics.INFIDELITY: InfidelityBatchComputationHandler,
    ExplanationMetrics.FAITHFULNESS_CORRELATION: FaithfulnessCorrelationBatchComputationHandler,
    ExplanationMetrics.FAITHFULNESS_ESTIMATE: FaithfulnessEstimateBatchComputationHandler,
    ExplanationMetrics.AOPC: AOPCBatchComputationHandler,
    ExplanationMetrics.MONOTONICITY: MonotonicityBatchComputationHandler,
    # robustness metrics
    ExplanationMetrics.SENSITIVITY: SensitivityBatchComputationHandler,
}


class BatchComputationHandlerFactory:
    @staticmethod
    def create_explanation_handler(
        output_file: Union[str, Path], **kwargs
    ) -> Type[ExplanationsBatchComputationHandler]:
        return ExplanationsBatchComputationHandler(output_file, **kwargs)

    @staticmethod
    def create_handler_for_metric(
        metric_name: str, output_file: Union[str, Path], **kwargs
    ) -> Explainer:
        handler_class = AVAILABLE_METRIC_HANDLERS.get(metric_name, None)
        if handler_class is None:
            raise ValueError(
                f"BatchComputationHandler for [{metric_name}] is not supported."
                f"Supported methods are: {AVAILABLE_METRIC_HANDLERS.keys()}."
            )
        return handler_class(output_file, **kwargs)
