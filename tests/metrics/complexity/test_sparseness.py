import logging
from logging import getLogger

import pytest  # noqa
import torch

from tests.utils.common import assert_tensor_almost_equal
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics import sparseness
from torchxai.metrics.complexity.sparseness import (
    _sparseness_feature_grouped,
    sparseness_feature_grouped,
)

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

test_configurations = [
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    TestRuntimeConfig(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([0.5501]),
    ),
    TestRuntimeConfig(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor([0.5308]),
    ),
    TestRuntimeConfig(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor([0.4295]),
    ),
    TestRuntimeConfig(
        test_name="basic_model_single_input_config_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.2500]),
    ),
    TestRuntimeConfig(
        test_name="basic_model_batch_input_config_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.2500] * 3),
    ),
    TestRuntimeConfig(
        test_name="basic_model_batch_input_with_additional_forward_args_config_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.0]),
    ),
    TestRuntimeConfig(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.3845] * 20),
    ),
    TestRuntimeConfig(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.2222, 0.0889, 0.0556, 0.0404]),
    ),
    TestRuntimeConfig(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.4444, 0.1111, 0.0635, 0.0444]),
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_sparseness(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration
    output = sparseness(
        attributions=explanations,
    )
    assert_tensor_almost_equal(
        output, runtime_config.expected, delta=runtime_config.delta, mode="mean"
    )


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_sparseness_feature_grouped(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration
    output = sparseness_feature_grouped(
        attributions=explanations,
    )
    assert_tensor_almost_equal(
        output, runtime_config.expected, delta=runtime_config.delta, mode="mean"
    )
