import dataclasses
import itertools

import pytest  # noqa
import torch

from tests.utils.common import (
    assert_all_tensors_almost_equal,
    assert_tensor_almost_equal,
    set_all_random_seeds,
)
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics import monotonicity
from torchxai.metrics._utils.common import _tuple_tensors_to_tensors


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    max_features_processed_per_batch: int = None
    device: str = "cpu"


test_configurations = [
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_random",
        target_fixture="park_function_configuration",
        explainer="random",
        expected=torch.tensor([False]),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([True]),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor([True]),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor([True]),
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_single_input_config_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        expected=torch.tensor([True]),
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_config_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        expected=torch.tensor([True] * 3),
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_with_additional_forward_args_config_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=torch.tensor([False]),
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=torch.tensor([True] * 20),
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([True] * 4),
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([True] * 4),
        max_features_processed_per_batch=[None, 1, 40],
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_monotonicity(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration

    runtime_config.max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )
    runtime_config.expected = _format_to_list(runtime_config.expected)

    assert (
        len(runtime_config.max_features_processed_per_batch)
        == len(runtime_config.expected)
        or len(runtime_config.expected) == 1
    )

    fwds_per_run = []
    for max_features, curr_expected in zip(
        runtime_config.max_features_processed_per_batch,
        itertools.cycle(runtime_config.expected),
    ):
        set_all_random_seeds(1234)
        (
            monotonicity_result,
            fwds,
        ) = monotonicity(
            forward_func=base_config.model,
            inputs=base_config.inputs,
            attributions=explanations,
            baselines=base_config.baselines,
            feature_mask=base_config.feature_mask,
            additional_forward_args=base_config.additional_forward_args,
            target=base_config.target,
            max_features_processed_per_batch=max_features,
        )
        assert_tensor_almost_equal(
            monotonicity_result.float(),
            curr_expected.float(),
            delta=runtime_config.delta,
            mode="mean",
        )
        explanations_flattened, _ = _tuple_tensors_to_tensors(explanations)
        assert len(fwds) == explanations_flattened.shape[0], (
            "The number of samples in the fwds must match the number of output explanations"
            "which is the same as the input batch size."
        )
        for fwd, explanation in zip(fwds, explanations_flattened):
            assert (
                fwd.numel() == explanation.numel()
            ), "The number of features should match the number of features in the explanations."  # match number of features
        fwds_per_run.append(torch.cat(fwds))
    assert_all_tensors_almost_equal(fwds_per_run)
