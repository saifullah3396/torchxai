import pytest  # noqa
import torch
from tests.utils.common import assert_tensor_almost_equal
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics.axiomatic.completeness import completeness

test_configurations = [
    TestRuntimeConfig(
        target_fixture="multi_modal_sequence_sum_with_single_target",
        explainer="integrated_gradients",
        expected=torch.tensor(
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
    ),
    TestRuntimeConfig(
        target_fixture="multi_modal_sequence_sum_with_single_target",
        explainer="saliency",
        expected=torch.tensor([116]),  # saliency completeness is not so great
    ),
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    TestRuntimeConfig(
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([1.6322]),  # saliency completeness is not so great
    ),
    TestRuntimeConfig(
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor(
            [0.1865]
        ),  # input_x_gradient results in better completeness
    ),
    TestRuntimeConfig(
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor(
            [1.3856e-08]
        ),  # integrated_gradients results in full completeness
    ),
    TestRuntimeConfig(
        target_fixture="basic_model_single_input_config",
        explainer="integrated_gradients",
        expected=torch.zeros(1),
    ),
    TestRuntimeConfig(
        target_fixture="basic_model_batch_input_config",
        explainer="integrated_gradients",
        expected=torch.zeros(3),
    ),
    TestRuntimeConfig(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        explainer="integrated_gradients",
        expected=torch.zeros(1),
    ),
    TestRuntimeConfig(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="deep_lift",
        expected=torch.zeros(20),
    ),
    TestRuntimeConfig(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([1.7565] * 20),
        delta=1e-3,
    ),
    TestRuntimeConfig(
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.6538, 0.0, 0.0, 0.0]),
    ),
    TestRuntimeConfig(
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        explainer="integrated_gradients",
        expected=torch.tensor([0.3269, 0.0, 0.0, 0.0]),
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_completeness(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration
    output = completeness(
        forward_func=base_config.model,
        inputs=base_config.inputs,
        attributions=explanations,
        metric_baselines=base_config.baselines,
        additional_forward_args=base_config.additional_forward_args,
        target=base_config.target,
    )
    assert_tensor_almost_equal(
        output, runtime_config.expected, delta=runtime_config.delta
    )
