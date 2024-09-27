import dataclasses
import itertools
from typing import Callable

import pytest  # noqa
import torch

from tests.utils.common import assert_tensor_almost_equal
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics._utils.perturbation import default_random_perturb_func
from torchxai.metrics.complexity.effective_complexity import effective_complexity


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    perturb_func: Callable = default_random_perturb_func()
    eps: float = 1e-5
    n_perturbations_per_feature: int = 10
    max_features_processed_per_batch: int = None


test_configurations = [
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([3]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
        eps=1e-2,
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor([3]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
        eps=1e-2,
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor([4]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
        eps=1e-2,
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_single_input_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        expected=torch.tensor([2]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        expected=torch.tensor([2, 2, 2]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_with_additional_forward_args_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=torch.tensor([0]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_deep_lift",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=torch.tensor([16] * 20),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_deep_lift_eps_1e-2",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=torch.tensor([14] * 20),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
        eps=1e-2,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_deep_lift_eps_1e-1",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=torch.tensor([10] * 20),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
        eps=1e-1,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([3, 3, 3, 3]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_integrated_gradients_eps_1e-1",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([1, 2, 2, 2]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
        eps=1e-1,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_integrated_gradients",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([3, 3, 3, 3]),
        n_perturbations_per_feature=[10, 10, 20],
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
def test_effective_complexity(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration

    runtime_config.n_perturbations_per_feature = _format_to_list(
        runtime_config.n_perturbations_per_feature
    )
    runtime_config.max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )
    runtime_config.expected = _format_to_list(runtime_config.expected)

    assert len(runtime_config.n_perturbations_per_feature) == len(
        runtime_config.max_features_processed_per_batch
    )
    assert (
        len(runtime_config.n_perturbations_per_feature) == len(runtime_config.expected)
        or len(runtime_config.expected) == 1
    )

    for n_perturbs, max_features, curr_expected in zip(
        runtime_config.n_perturbations_per_feature,
        runtime_config.max_features_processed_per_batch,
        itertools.cycle(runtime_config.expected),
    ):
        (
            effective_complexity_score,
            k_features_perturbed_fwd_diff_vars,
            n_features_found,
        ) = effective_complexity(
            forward_func=base_config.model,
            inputs=base_config.inputs,
            attributions=explanations,
            feature_mask=base_config.feature_mask,
            additional_forward_args=base_config.additional_forward_args,
            target=base_config.target,
            perturb_func=runtime_config.perturb_func,
            n_perturbations_per_feature=n_perturbs,
            max_features_processed_per_batch=max_features,
            eps=runtime_config.eps,
            use_absolute_attributions=True,
        )
        assert_tensor_almost_equal(
            effective_complexity_score, curr_expected, delta=runtime_config.delta
        )
        assert (
            n_features_found[0].item() == base_config.n_features
        ), f"{n_features_found} != {base_config.n_features}"
