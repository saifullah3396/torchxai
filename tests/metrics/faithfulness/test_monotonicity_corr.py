import dataclasses
import itertools
from typing import Callable

import pytest  # noqa
import torch

from tests.utils.common import assert_tensor_almost_equal, set_all_random_seeds
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics import monotonicity_corr_and_non_sens
from torchxai.metrics._utils.perturbation import default_random_perturb_func


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    perturb_func: Callable = default_random_perturb_func()
    n_perturbations_per_feature: int = 100
    max_features_processed_per_batch: int = None


test_configurations = [
    # these monotonicity results match from the paper: https://arxiv.org/pdf/2007.07584
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([0.9852]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor([0.8235]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor([0.5882]),
        perturb_func=default_random_perturb_func(noise_scale=1.0),
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_single_input_config_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        expected=torch.ones(1),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_config_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        expected=torch.ones(3),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_with_additional_forward_args_config_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=torch.tensor([torch.nan]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            torch.tensor(
                [
                    0.3410,
                    0.3793,
                    0.3410,
                    0.3658,
                    0.3422,
                    0.3646,
                    0.3410,
                    0.3658,
                    0.3658,
                    0.3422,
                    0.3557,
                    0.3746,
                    0.3422,
                    0.3628,
                    0.3776,
                    0.3540,
                    0.3776,
                    0.3422,
                    0.3557,
                    0.3422,
                ],
            ),
            torch.tensor(
                [
                    0.3410,
                    0.3793,
                    0.3410,
                    0.3658,
                    0.3422,
                    0.3646,
                    0.3410,
                    0.3658,
                    0.3658,
                    0.3422,
                    0.3557,
                    0.3746,
                    0.3422,
                    0.3628,
                    0.3776,
                    0.3540,
                    0.3776,
                    0.3422,
                    0.3557,
                    0.3422,
                ],
            ),
            torch.tensor(
                [
                    0.3569,
                    0.3540,
                    0.3569,
                    0.3805,
                    0.3540,
                    0.3658,
                    0.3776,
                    0.3422,
                    0.3569,
                    0.3557,
                    0.3439,
                    0.3628,
                    0.3658,
                    0.3658,
                    0.3658,
                    0.3658,
                    0.3540,
                    0.3805,
                    0.3569,
                    0.3658,
                ],
            ),
        ],
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
        delta=1e-3,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([1, 1, 1, 1]),
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([1, 1, 1, 1]),
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
        set_all_random_seeds(1234)
        (
            monotonicity_corr_score,
            _,
            n_features_found,
        ) = monotonicity_corr_and_non_sens(
            forward_func=base_config.model,
            inputs=base_config.inputs,
            attributions=explanations,
            feature_mask=base_config.feature_mask,
            additional_forward_args=base_config.additional_forward_args,
            target=base_config.target,
            perturb_func=runtime_config.perturb_func,
            n_perturbations_per_feature=n_perturbs,
            max_features_processed_per_batch=max_features,
        )
        assert_tensor_almost_equal(
            monotonicity_corr_score, curr_expected, delta=runtime_config.delta
        )
        assert (
            n_features_found[0].item() == base_config.n_features
        ), f"{n_features_found} != {base_config.n_features}"
