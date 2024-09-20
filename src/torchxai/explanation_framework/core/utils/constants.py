from enum import Enum


class EmbeddingKeys(str, Enum):
    TOKEN_EMBEDDINGS = "token_embeddings"
    POSITION_EMBEDDINGS = "position_embeddings"
    SPATIAL_POSITION_EMBEDDINGS = "spatial_embeddings"
    PATCH_EMBEDDINGS = "patch_embeddings"
    TOKEN_TYPE_EMBEDDINGS = "token_type_embeddings"


class ExplanationMetrics(str, Enum):
    # axiomatic
    COMPLETENESS = "completeness"
    MONOTONICITY_CORR_AND_NON_SENS = "monotonicity_corr_and_non_sens"

    # complexity
    EFFECTIVE_COMPLEXITY = "effective_complexity"
    COMPLEXITY = "complexity"
    SPARSENESS = "sparseness"

    # faithfulness
    FAITHFULNESS_CORR = "faithfulness_corr"
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
