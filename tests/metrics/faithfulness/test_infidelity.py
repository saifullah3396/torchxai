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
from torchxai.metrics import infidelity


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


test_configurations = [
    # the park function is taken from the paper: https://arxiv.org/pdf/2007.07584
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_random",
        target_fixture="park_function_configuration",
        explainer="random",
        expected=torch.tensor([0.0399]),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_saliency",
        target_fixture="park_function_configuration",
        explainer="saliency",
        expected=torch.tensor([0.0003]),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_input_x_gradient",
        target_fixture="park_function_configuration",
        explainer="input_x_gradient",
        expected=torch.tensor([0.0032]),
    ),
    MetricTestRuntimeConfig_(
        test_name="park_function_configuration_integrated_gradients",
        target_fixture="park_function_configuration",
        explainer="integrated_gradients",
        expected=torch.tensor([0.0043]),
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_single_input_config_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        expected=[
            torch.tensor([0.0015]),
            # changing the batch size changes the output since perturbation is random generated based on input shape
            # which is different when batch size is different
            torch.tensor([0.0021]),
            torch.tensor([0.0015]),
        ],
        max_examples_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_config_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        expected=[
            torch.tensor([0.0031, 0.0033, 0.0032]),
            # changing the batch size changes the output since perturbation is random generated based on input shape
            # which is different when batch size is different
            torch.tensor([0.0019, 0.0045, 0.0042]),
            torch.tensor([0.0031, 0.0033, 0.0032]),
        ],
        max_examples_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_with_additional_forward_args_config_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=torch.tensor([0]),
        max_examples_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            torch.tensor(
                [
                    0.1320,
                    0.1129,
                    0.1996,
                    0.1540,
                    0.2256,
                    0.3201,
                    0.2720,
                    0.1503,
                    0.1376,
                    0.1383,
                    0.0667,
                    0.3166,
                    0.1227,
                    0.0961,
                    0.0932,
                    0.2056,
                    0.1310,
                    0.2620,
                    0.1342,
                    0.1308,
                ]
            ),
            torch.tensor(
                [
                    0.1077,
                    0.1544,
                    0.2815,
                    0.0737,
                    0.0831,
                    0.1985,
                    0.1400,
                    0.1247,
                    0.1435,
                    0.2103,
                    0.1101,
                    0.2772,
                    0.2653,
                    0.1276,
                    0.2102,
                    0.0717,
                    0.2273,
                    0.2970,
                    0.1900,
                    0.1480,
                ]
            ),
            torch.tensor(
                [
                    0.1052,
                    0.2192,
                    0.0722,
                    0.1192,
                    0.2457,
                    0.1706,
                    0.2629,
                    0.1006,
                    0.2482,
                    0.1780,
                    0.1603,
                    0.1364,
                    0.2134,
                    0.1490,
                    0.1112,
                    0.2140,
                    0.1223,
                    0.2354,
                    0.2309,
                    0.1758,
                ]
            ),
        ],
        max_examples_per_batch=[None, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=[
            torch.tensor([2.0656, 1.4134, 0.1787, 0.1492]),
            torch.tensor([3.7065, 0.2863, 0.0823, 0.2374]),
            torch.tensor([2.0656, 1.4134, 0.1787, 0.1492]),
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
def test_infidelity(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration

    runtime_config.max_examples_per_batch = _format_to_list(
        runtime_config.max_examples_per_batch
    )
    runtime_config.expected = _format_to_list(runtime_config.expected)

    assert (
        len(runtime_config.max_examples_per_batch) == len(runtime_config.expected)
        or len(runtime_config.expected) == 1
    )

    def perturb_fn(inputs, baselines=None, feature_masks=None, frozen_features=None):
        is_input_tuple = isinstance(inputs, tuple)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        noise = tuple(
            torch.randn_like(
                x,
                device=x.device,
            )
            * 0.1
            for x in inputs
        )

        if is_input_tuple:
            return noise, tuple(x - y for x, y in zip(inputs, noise))
        else:
            return noise[0], inputs[0] - noise[0]

    fwds_per_run = []
    for max_examples_per_batch, curr_expected in zip(
        runtime_config.max_examples_per_batch,
        itertools.cycle(runtime_config.expected),
    ):
        set_all_random_seeds(1234)
        infidelity_result = infidelity(
            forward_func=base_config.model,
            perturb_func=perturb_fn,
            inputs=base_config.inputs,
            attributions=explanations,
            baselines=base_config.baselines,
            additional_forward_args=base_config.additional_forward_args,
            target=base_config.target,
            n_perturb_samples=runtime_config.n_perturb_samples,
            max_examples_per_batch=max_examples_per_batch,
            normalize=runtime_config.normalize,
        )
        assert_tensor_almost_equal(
            infidelity_result.float(),
            curr_expected.float(),
            delta=runtime_config.delta,
            mode="mean",
        )
    assert_all_tensors_almost_equal(fwds_per_run)
