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
from torchxai.metrics import sensitivity_n


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    n_perturb_samples: int = 10
    max_examples_per_batch: int = None
    normalize: bool = True
    device: str = "cpu"
    sensitivity_n: int = 1


def generate_park_function_configurations(
    test_name, target_fixture, explainer, expected_for_each_n, sensitivity_n
):
    return [
        MetricTestRuntimeConfig_(
            test_name=test_name,
            target_fixture=target_fixture,
            explainer=explainer,
            expected=torch.tensor([expected]),
            sensitivity_n=n,
        )
        for expected, n in zip(expected_for_each_n, sensitivity_n)
    ]


test_configurations = [
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    *generate_park_function_configurations(
        test_name="park_function_configuration_random",
        target_fixture="park_function_configuration",
        explainer="random",
        expected_for_each_n=[0.1254, 0.0902, 0.0931],
        sensitivity_n=[1, 2, 3],
    ),
    *generate_park_function_configurations(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected_for_each_n=[0.0567, 0.0567, 0.0259],
        sensitivity_n=[1, 2, 3],
    ),
    *generate_park_function_configurations(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected_for_each_n=[0.0006, 0.0301, 0.0551],
        sensitivity_n=[1, 2, 3],
    ),
    *generate_park_function_configurations(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected_for_each_n=[0.0244, 0.0337, 0.0106],
        sensitivity_n=[1, 2, 3],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_single_input_config_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        expected=torch.tensor([0.2286]),
        max_examples_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_config_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        expected=[
            torch.tensor([0.2286, 0.1273, 0.1273]),
            torch.tensor([0.2000, 0.1273, 0.1655]),
            torch.tensor([0.2286, 0.1273, 0.1273]),
        ],
        max_examples_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_with_additional_forward_args_config_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=torch.tensor([0.1000]),
        max_examples_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            torch.tensor(
                [
                    31.8245,
                    9.4489,
                    19.6486,
                    22.2977,
                    18.9699,
                    23.0527,
                    8.8643,
                    19.5944,
                    25.1141,
                    26.6413,
                    7.7459,
                    22.0229,
                    34.0302,
                    27.1363,
                    20.0385,
                    37.6928,
                    18.6043,
                    19.3381,
                    25.0242,
                    21.2561,
                ]
            ),
            torch.tensor(
                [
                    21.6524,
                    24.9081,
                    13.7743,
                    4.3087,
                    28.9788,
                    18.0455,
                    19.0464,
                    35.0910,
                    8.0913,
                    21.0251,
                    29.8616,
                    17.1790,
                    21.4008,
                    16.0830,
                    34.5897,
                    26.0547,
                    23.9150,
                    24.1518,
                    26.2766,
                    9.5782,
                ]
            ),
            torch.tensor(
                [
                    22.2922,
                    14.0161,
                    23.2412,
                    33.1589,
                    13.0975,
                    19.6891,
                    13.8579,
                    21.6143,
                    20.1616,
                    21.0284,
                    5.3459,
                    7.9667,
                    20.6926,
                    18.4868,
                    22.1537,
                    26.2134,
                    25.2676,
                    38.0853,
                    27.9821,
                    11.3850,
                ]
            ),
        ],
        max_examples_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=[
            torch.tensor([4.0000e00, 0.0000e00, 1.1642e-11, -2.3283e-11]),
            torch.tensor([4.8302e00, 0.0000e00, 1.1642e-11, -2.3283e-11]),
            torch.tensor([4.0000e00, 0.0000e00, 1.1642e-11, -2.3283e-11]),
        ],
        max_examples_per_batch=[None, 1, 40],
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_sensitivity_n(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration

    runtime_config.max_examples_per_batch = _format_to_list(
        runtime_config.max_examples_per_batch
    )
    runtime_config.expected = _format_to_list(runtime_config.expected)

    assert (
        len(runtime_config.max_examples_per_batch) == len(runtime_config.expected)
        or len(runtime_config.expected) == 1
    )

    fwds_per_run = []
    for max_examples_per_batch, curr_expected in zip(
        runtime_config.max_examples_per_batch,
        itertools.cycle(runtime_config.expected),
    ):
        set_all_random_seeds(1234)
        sensitivity_n_result = sensitivity_n(
            forward_func=base_config.model,
            inputs=base_config.inputs,
            attributions=explanations,
            baselines=base_config.baselines if base_config.baselines is not None else 0,
            n_features_perturbed=runtime_config.sensitivity_n,
            additional_forward_args=base_config.additional_forward_args,
            target=base_config.target,
            n_perturb_samples=runtime_config.n_perturb_samples,
            max_examples_per_batch=max_examples_per_batch,
            normalize=runtime_config.normalize,
        )
        assert_tensor_almost_equal(
            sensitivity_n_result.float(),
            curr_expected.float(),
            delta=runtime_config.delta,
            mode="mean",
        )
    assert_all_tensors_almost_equal(fwds_per_run)
