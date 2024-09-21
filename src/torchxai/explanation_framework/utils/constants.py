from enum import Enum


class ExplanationMetrics(str, Enum):
    # axiomatic
    COMPLETENESS = "completeness"
    MONOTONICITY_CORR_AND_NON_SENS = "monotonicity_corr_and_non_sens"

    # complexity
    EFFECTIVE_COMPLEXITY = "effective_complexity"
    COMPLEXITY = "complexity"
    SPARSENESS = "sparseness"

    # faithfulness
    FAITHFULNESS_CORRELATION = "faithfulness_corr"
    FAITHFULNESS_ESTIMATE = "faithfulness_estimate"
    MONOTONICITY = "monotonicity"
    INFIDELITY = "infidelity"
    AOPC = "aopc"

    # robustness
    SENSITIVITY = "sensitivity"


RAW_EXPLANATION_DEPENDENT_METRICS = [
    ExplanationMetrics.INFIDELITY,
    ExplanationMetrics.COMPLETENESS,
    ExplanationMetrics.MONOTONICITY_CORR_AND_NON_SENS,
]
