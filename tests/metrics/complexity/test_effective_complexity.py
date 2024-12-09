import dataclasses
import itertools
import math
from typing import Callable

import pytest  # noqa
import torch

from tests.utils.common import assert_tensor_almost_equal
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics._utils.perturbation import (
    default_fixed_baseline_perturb_func,
    default_random_perturb_func,
)
from torchxai.metrics.complexity.effective_complexity import effective_complexity


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    perturb_func: Callable = default_random_perturb_func()
    zero_variance_threshold: float = 1e-5
    n_perturbations_per_feature: int = 10
    max_features_processed_per_batch: int = None
    percentage_feature_removal_per_step: float = 0.0
    return_ratio: bool = False


test_configurations = [
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 4 features have variance above 0.01
            [0.1481]
        ),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        zero_variance_threshold=0.01,
        return_ratio=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 4 features have variance above 0.01
            [0.1481]
        ),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        zero_variance_threshold=1e-4,
        return_ratio=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 4 features have variance above 0.01
            [0.2222]
        ),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        zero_variance_threshold=1e-4,
        percentage_feature_removal_per_step=0.1,
        return_ratio=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 4 features have variance above 0.01
            [0.2222]
        ),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        zero_variance_threshold=1e-4,
        percentage_feature_removal_per_step=0.1,
        return_ratio=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor([0.4074]),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        zero_variance_threshold=1e-1,
        return_ratio=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10",
        target_fixture="multi_modal_sequence_sum",
        explainer="saliency",
        expected=torch.tensor([0.4074]),  # saliency completeness is not so great
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        zero_variance_threshold=1e-1,
        return_ratio=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor([0.4444]),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        zero_variance_threshold=1e-1,
        percentage_feature_removal_per_step=0.1,
        return_ratio=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10_percentage_feature_removal_per_step_0.1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor([0.4444]),
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        zero_variance_threshold=1e-1,
        percentage_feature_removal_per_step=0.1,
        return_ratio=True,
    ),
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([3]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
        zero_variance_threshold=1e-2,
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor([3]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
        zero_variance_threshold=1e-2,
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor([4]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
        zero_variance_threshold=1e-2,
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
        test_name="classification_convnet_model_with_multiple_targets_deep_lift_zero_variance_threshold_1e-2",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=torch.tensor([14] * 20),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
        zero_variance_threshold=1e-2,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_deep_lift_zero_variance_threshold_1e-1",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=torch.tensor([10] * 20),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
        zero_variance_threshold=1e-1,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([3, 3, 3, 3]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_integrated_gradients_zero_variance_threshold_1e-1",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([1, 2, 2, 2]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
        zero_variance_threshold=1e-1,
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
            baselines=base_config.baselines,
            feature_mask=base_config.feature_mask,
            additional_forward_args=base_config.additional_forward_args,
            target=base_config.target,
            perturb_func=runtime_config.perturb_func,
            n_perturbations_per_feature=n_perturbs,
            max_features_processed_per_batch=max_features,
            percentage_feature_removal_per_step=runtime_config.percentage_feature_removal_per_step,
            frozen_features=base_config.frozen_features,
            zero_variance_threshold=runtime_config.zero_variance_threshold,
            return_ratio=runtime_config.return_ratio,
            return_intermediate_results=True,
            show_progress=True,
        )
        target_n_features = (
            base_config.n_features
            if runtime_config.percentage_feature_removal_per_step == 0.0
            else base_config.n_features
            // math.ceil(
                base_config.n_features
                * runtime_config.percentage_feature_removal_per_step
            )
        )
        assert_tensor_almost_equal(
            effective_complexity_score, curr_expected, delta=runtime_config.delta
        )
        assert (
            n_features_found[0] == target_n_features
        ), f"{n_features_found[0]} != {target_n_features}"
