from dataclasses import dataclass, field
from typing import List

import pytest
import torch  # noqa

from tests.utils.common import assert_tensor_almost_equal
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics import complexity


@dataclass
class MetricTestRuntimeConfig(TestRuntimeConfig):
    test_name: str = "compare_multi_target_to_single_target"
    explainer: str = "saliency"
    override_target: List[int] = field(default_factory=lambda: [0, 1, 2])
    expected: torch.Tensor = None
    explainer_kwargs: dict = field(default_factory=lambda: {"is_multi_target": True})
    delta: float = 1e-8


test_configurations = [
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_config",
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_single_sample_config",
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_real_images_config",
        explainer="integrated_gradients",
    ),
    MetricTestRuntimeConfig(
        target_fixture="classification_alexnet_model_real_images_single_sample_config",
        explainer="integrated_gradients",
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_complexity_multi_target(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration
    assert len(explanations) == len(
        runtime_config.override_target
    ), "Number of explanations should be equal to the number of targets"

    per_target_complexity = []
    for explanation, target in zip(explanations, runtime_config.override_target):
        output = complexity(
            attributions=explanation,
        )
        per_target_complexity.append(output)

    multi_target_complexity_output = complexity(
        attributions=explanations,
        is_multi_target=True,
    )

    assert len(per_target_complexity) == len(multi_target_complexity_output)
    for output, expected in zip(multi_target_complexity_output, per_target_complexity):
        assert_tensor_almost_equal(output, expected, delta=runtime_config.delta)
