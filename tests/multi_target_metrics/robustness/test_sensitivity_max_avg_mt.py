import inspect
from dataclasses import dataclass, field
from typing import List

import pytest
import torch  # noqa

from tests.utils.common import assert_tensor_almost_equal, set_all_random_seeds
from tests.utils.containers import RuntimeTestConfig
from torchxai.metrics import sensitivity_max_and_avg


@dataclass
class MetricTestRuntimeConfig(RuntimeTestConfig):
    test_name: str = "compare_multi_target_to_single_target"
    explainer: str = "saliency"
    override_target: List[int] = field(default_factory=lambda: [0, 1, 2])
    expected: torch.Tensor = None
    explainer_kwargs: dict = field(default_factory=lambda: {"is_multi_target": True})
    delta: float = 1e-5


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
    "explainer_metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_completeness_multi_target(explainer_metrics_runtime_test_configuration):
    base_config, runtime_config, explainer = (
        explainer_metrics_runtime_test_configuration
    )

    explainer_kwargs = {}
    possible_args = inspect.signature(explainer.explain).parameters
    if base_config.baselines is not None and "baselines" in possible_args:
        explainer_kwargs["explainer_baselines"] = base_config.baselines
    if base_config.feature_mask is not None and "feature_mask" in possible_args:
        explainer_kwargs["feature_mask"] = base_config.feature_mask
    if base_config.train_baselines is not None and "train_baselines" in possible_args:
        explainer_kwargs["train_baselines"] = base_config.train_baselines

    per_target_sensitivity_max = []
    per_target_sensitivity_avg = []
    for target in runtime_config.override_target:
        explainer.is_multi_target = False
        set_all_random_seeds(1234)
        sens_max, sens_avg = sensitivity_max_and_avg(
            explainer=explainer,
            inputs=base_config.inputs,
            additional_forward_args=base_config.additional_forward_args,
            target=target,
            **explainer_kwargs,
        )
        per_target_sensitivity_max.append(sens_max)
        per_target_sensitivity_avg.append(sens_avg)

    explainer.is_multi_target = True
    set_all_random_seeds(1234)
    multi_target_sens_max, multi_target_sens_avg = sensitivity_max_and_avg(
        explainer=explainer,
        inputs=base_config.inputs,
        additional_forward_args=base_config.additional_forward_args,
        target=runtime_config.override_target,
        is_multi_target=True,
        **explainer_kwargs,
    )
    assert len(per_target_sensitivity_max) == len(multi_target_sens_max)
    assert len(per_target_sensitivity_avg) == len(multi_target_sens_avg)
    for output, expected in zip(multi_target_sens_max, per_target_sensitivity_max):
        assert_tensor_almost_equal(
            output, expected, delta=runtime_config.delta, mode="mean"
        )
    for output, expected in zip(multi_target_sens_avg, per_target_sensitivity_avg):
        assert_tensor_almost_equal(
            output, expected, delta=runtime_config.delta, mode="mean"
        )
