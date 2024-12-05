import dataclasses
import itertools
import math
from typing import Callable

import pytest  # noqa
import torch

from tests.utils.common import assert_tensor_almost_equal
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics import monotonicity_corr_and_non_sens
from torchxai.metrics._utils.perturbation import (
    default_fixed_baseline_perturb_func,
    default_random_perturb_func,
)


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    perturb_func: Callable = default_random_perturb_func()
    n_perturbations_per_feature: int = 100
    max_features_processed_per_batch: int = None
    zero_attribution_threshold: float = 1e-5
    zero_variance_threshold: float = 1e-5
    use_percentage_attribution_threshold: bool = False
    feature_removal_beam_size_percentage: float = 0.0


test_configurations = [
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1_feature_removal_beam_size_percentage_0.1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
        feature_removal_beam_size_percentage=0.1,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10_feature_removal_beam_size_percentage_0.1",
        target_fixture="multi_modal_sequence_relu",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
        feature_removal_beam_size_percentage=0.1,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10",
        target_fixture="multi_modal_sequence_sum",
        explainer="saliency",
        # in this test set first 5 features have variance below 1e-4 and
        # and all attributions are the same as importance is assigned 1 to all features by Saliency
        expected=torch.tensor([5]),  # saliency completeness is not so great
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_1_feature_removal_beam_size_percentage_0.1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=1,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
        feature_removal_beam_size_percentage=0.1,
    ),
    MetricTestRuntimeConfig_(
        test_name="n_perturbations_per_feature_10_feature_removal_beam_size_percentage_0.1",
        target_fixture="multi_modal_sequence_sum",
        explainer="integrated_gradients",
        expected=torch.tensor(
            # in this test set first 5 features have variance below 1e-4 and
            # and first first attributions have values below 0.01
            [0.0]
        ),  # integrated gradients completeness should be 0 for this case
        perturb_func=default_fixed_baseline_perturb_func(),
        n_perturbations_per_feature=10,
        use_percentage_attribution_threshold=False,
        zero_variance_threshold=1e-4,
        zero_attribution_threshold=0.01,
        feature_removal_beam_size_percentage=0.1,
    ),
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([0]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor([0]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor([0]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_single_input_config_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        expected=torch.zeros(1),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_config_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        expected=torch.zeros(3),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_with_additional_forward_args_config_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=torch.ones(1),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=torch.tensor([4] * 20),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([1, 0, 0, 0]),
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
def test_non_sensitivity(metrics_runtime_test_configuration):
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
            _,
            non_sensitivity_score,
            n_features_found,
            _,
            _,
        ) = monotonicity_corr_and_non_sens(
            forward_func=base_config.model,
            inputs=base_config.inputs,
            attributions=explanations,
            feature_mask=base_config.feature_mask,
            baselines=base_config.baselines,
            additional_forward_args=base_config.additional_forward_args,
            target=base_config.target,
            frozen_features=base_config.frozen_features,
            perturb_func=runtime_config.perturb_func,
            n_perturbations_per_feature=n_perturbs,
            max_features_processed_per_batch=max_features,
            zero_attribution_threshold=runtime_config.zero_attribution_threshold,
            zero_variance_threshold=runtime_config.zero_variance_threshold,
            use_percentage_attribution_threshold=runtime_config.use_percentage_attribution_threshold,
            feature_removal_beam_size_percentage=runtime_config.feature_removal_beam_size_percentage,
            return_intermediate_results=True,
            return_ratio=False,
        )
        assert_tensor_almost_equal(
            non_sensitivity_score, curr_expected, delta=runtime_config.delta
        )
        target_n_features = (
            base_config.n_features
            if runtime_config.feature_removal_beam_size_percentage == 0.0
            else base_config.n_features
            // math.ceil(
                base_config.n_features
                * runtime_config.feature_removal_beam_size_percentage
            )
        )
        assert (
            n_features_found[0].item() == target_n_features
        ), f"{n_features_found} != {target_n_features}"
