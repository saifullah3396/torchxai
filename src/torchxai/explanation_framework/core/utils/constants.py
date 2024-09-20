class EMBEDDING_KEYS:
    TOKEN_EMBEDDINGS = "token_embeddings"
    POSITION_EMBEDDINGS = "position_embeddings"
    SPATIAL_POSITION_EMBEDDINGS = "spatial_embeddings"
    PATCH_EMBEDDINGS = "patch_embeddings"
    TOKEN_TYPE_EMBEDDINGS = "token_type_embeddings"


class EXPLANATION_METRICS:
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
    EXPLANATION_METRICS.INFIDELITY,
    EXPLANATION_METRICS.COMPLETENESS,
    EXPLANATION_METRICS.MONOTONICITY_CORR_AND_NON_SENS,
]
