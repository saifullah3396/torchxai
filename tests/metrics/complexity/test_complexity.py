import logging
from logging import getLogger

import pytest  # noqa
import torch

from tests.utils.common import assert_tensor_almost_equal
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics.complexity.complexity import complexity

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

test_configurations = [
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    TestRuntimeConfig(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([1.1656]),
    ),
    TestRuntimeConfig(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor([1.2182]),
    ),
    TestRuntimeConfig(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor([1.3492]),
    ),
    TestRuntimeConfig(
        test_name="basic_model_single_input_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.5623]),
    ),
    TestRuntimeConfig(
        test_name="basic_model_batch_input_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.5623] * 3),
    ),
    TestRuntimeConfig(
        test_name="basic_model_batch_input_with_additional_forward_args_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        explainer="integrated_gradients",
        expected=torch.tensor([1.7918]),
    ),
    TestRuntimeConfig(
        test_name="classification_convnet_model_with_multiple_targets_integrated_gradients",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([2.5078] * 20),
    ),
    TestRuntimeConfig(
        test_name="classification_multilayer_model_with_tuple_targets_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([1.0114, 1.0852, 1.0934, 1.0959]),
    ),
    TestRuntimeConfig(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_integrated_gradients",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.6365, 1.0776, 1.0918, 1.0953]),
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_complexity(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration
    output = complexity(
        attributions=explanations,
    )
    assert_tensor_almost_equal(
        output, runtime_config.expected, delta=runtime_config.delta, mode="mean"
    )
