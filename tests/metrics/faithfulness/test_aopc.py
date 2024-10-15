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
from torchxai.metrics import aopc
from torchxai.metrics._utils.common import _tuple_tensors_to_tensors


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    max_features_processed_per_batch: int = None
    total_features_perturbed: int = 100
    expected_desc: torch.Tensor = None
    expected_asc: torch.Tensor = None
    expected_rand: torch.Tensor = None

    def __post_init__(self):
        assert self.target_fixture is not None, "Target fixture must be provided"
        assert (
            self.expected_desc is not None
        ), "Expected value for aopc_desc must be provided"
        assert (
            self.expected_asc is not None
        ), "Expected value for aopc_desc must be provided"
        assert (
            self.expected_rand is not None
        ), "Expected value for aopc_desc must be provided"
        if self.explainer_kwargs is None:
            self.explainer_kwargs = {}


test_configurations = [
    MetricTestRuntimeConfig_(
        test_name="basic_model_single_input_config_integrated_gradients",
        target_fixture="basic_model_single_input_config",
        expected_desc=torch.tensor([[0.0000, 0.5000, 0.6667]]).unbind(),
        expected_asc=torch.tensor([[0.0000, -0.5000, 0.0000]]).unbind(),
        expected_rand=torch.tensor([[0.0000, 0.2000, 0.4667]]).unbind(),
        max_features_processed_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_config_integrated_gradients",
        target_fixture="basic_model_batch_input_config",
        expected_desc=torch.tensor([[0.0000, 0.5000, 0.6667]] * 3).unbind(),
        expected_asc=torch.tensor([[0.0000, -0.5000, 0.0000]] * 3).unbind(),
        expected_rand=torch.tensor([[0.0000, 0.2000, 0.4667]] * 3).unbind(),
        max_features_processed_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_with_additional_forward_args_config_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected_desc=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).unbind(),
        expected_asc=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).unbind(),
        expected_rand=torch.tensor(
            [[0.0000, 0.0000, -0.0167, -0.0500, -0.0800, -0.0750, -0.0643]]
        ).unbind(),
        max_features_processed_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected_desc=torch.tensor(
            [
                [
                    0.0000,
                    7.5000,
                    12.6667,
                    22.0000,
                    31.0000,
                    37.0000,
                    41.2857,
                    44.5000,
                    47.0000,
                    49.0000,
                    50.6364,
                    52.0000,
                    53.1538,
                    54.1429,
                    55.0000,
                    55.7500,
                    56.4118,
                ]
            ]
            * 20
        ).unbind(),
        expected_asc=torch.tensor(
            [
                [
                    0.0000,
                    0.5000,
                    2.0000,
                    3.7500,
                    5.6000,
                    6.8333,
                    7.7143,
                    9.8750,
                    12.0000,
                    13.7000,
                    16.1818,
                    19.2500,
                    22.9231,
                    26.0714,
                    28.8000,
                    31.1875,
                    33.2941,
                ]
            ]
            * 20
        ).unbind(),
        expected_rand=torch.tensor(
            [
                [
                    0.0000,
                    3.1500,
                    5.4000,
                    7.6750,
                    9.6800,
                    12.5333,
                    16.1000,
                    19.3625,
                    22.8000,
                    26.2400,
                    29.5727,
                    32.5833,
                    35.2308,
                    37.5000,
                    39.4667,
                    41.1875,
                    42.7059,
                ]
            ]
            * 20
        ).unbind(),
        max_features_processed_per_batch=[5, 1, 40],
        delta=1e-3,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected_desc=torch.tensor(
            [
                [0.0000, 44.0000, 74.6667, 96.0000],
                [0.0000, 96.0000, 181.3333, 256.0000],
                [0.0000, 144.0000, 277.3333, 400.0000],
                [0.0000, 192.0000, 373.3333, 544.0000],
            ]
        ).unbind(),
        expected_asc=torch.tensor(
            [
                [0.0000, 16.0000, 40.0000, 70.0000],
                [0.0000, 64.0000, 138.6667, 224.0000],
                [0.0000, 112.0000, 234.6667, 368.0000],
                [0.0000, 160.0000, 330.6667, 512.0000],
            ]
        ).unbind(),
        expected_rand=torch.tensor(
            [
                [0.0000, 28.0000, 53.6000, 80.2000],
                [0.0000, 76.8000, 154.6667, 236.0000],
                [0.0000, 124.8000, 250.6667, 380.0000],
                [0.0000, 172.8000, 346.6667, 524.0000],
            ]
        ).unbind(),
        max_features_processed_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected_desc=torch.tensor(
            [
                [0.0000, 32.0000, 50.6667, 60.0000],
                [0.0000, 80.0000, 149.3333, 208.0000],
                [0.0000, 128.0000, 245.3333, 352.0000],
                [0.0000, 176.0000, 341.3333, 496.0000],
            ]
        ).unbind(),
        expected_asc=torch.tensor(
            [
                [0.0000, 0.0000, 10.6667, 30.0000],
                [0.0000, 48.0000, 106.6667, 176.0000],
                [0.0000, 96.0000, 202.6667, 320.0000],
                [0.0000, 144.0000, 298.6667, 464.0000],
            ]
        ).unbind(),
        expected_rand=torch.tensor(
            [
                [0.0000, 12.8000, 26.1333, 41.6000],
                [0.0000, 60.8000, 122.6667, 188.0000],
                [0.0000, 108.8000, 218.6667, 332.0000],
                [0.0000, 156.8000, 314.6667, 476.0000],
            ]
        ).unbind(),
        max_features_processed_per_batch=[5, 1, 40],
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
    runtime_config.max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )
    runtime_config.expected_desc = _format_to_list(runtime_config.expected_desc)
    runtime_config.expected_asc = _format_to_list(runtime_config.expected_asc)
    runtime_config.expected_rand = _format_to_list(runtime_config.expected_rand)

    assert (
        len(runtime_config.max_features_processed_per_batch)
        == len(runtime_config.expected_desc)
        or len(runtime_config.expected_desc) == 1
    )
    assert (
        len(runtime_config.expected_desc)
        == len(runtime_config.expected_asc)
        == len(runtime_config.expected_rand)
    )

    aopcs_desc_list = []
    aopcs_asc_list = []
    aopcs_rand_list = []
    for max_features, curr_expected_desc, curr_expected_asc, curr_expected_rand in zip(
        runtime_config.max_features_processed_per_batch,
        itertools.cycle(runtime_config.expected_desc),
        itertools.cycle(runtime_config.expected_asc),
        itertools.cycle(runtime_config.expected_rand),
    ):
        set_all_random_seeds(1234)
        aopc_output = aopc(
            forward_func=base_config.model,
            inputs=base_config.inputs,
            attributions=explanations,
            baselines=base_config.baselines,
            feature_mask=base_config.feature_mask,
            additional_forward_args=base_config.additional_forward_args,
            target=base_config.target,
            max_features_processed_per_batch=max_features,
            total_features_perturbed=runtime_config.total_features_perturbed,
            seed=42,
        )
        explanations_flattened, _ = _tuple_tensors_to_tensors(explanations)

        for x in [aopc_output.desc, aopc_output.asc, aopc_output.rand]:
            # match the batch size
            assert len(x) == explanations_flattened.shape[0], (
                f"The number of samples in the aopc output should match the number of samples in the input. "
                f"Expected: {explanations_flattened.shape[0]}, Got: {len(x)}"
            )

            # total output size is number of features + 1 since the fwd with no perturbation is also included
            # either that or maxed at total_features_perturbed
            # current tests assume all inputs have the same number of features
            # these will fail in case a mask is used and the number of features is different for each input
            # in the batch
            expected_output_size = min(
                runtime_config.total_features_perturbed,
                explanations_flattened[0].numel(),
            )

            # current tests assume all inputs have the same number of features
            # these will fail in case a mask is used and the number of features is different for each input
            # in the batch
            assert (
                x[0].shape[0] == expected_output_size + 1
            ), f"The output size of aopcs is invalid. Expected: {expected_output_size}, Got: {x[0].shape[0]}"

        for output, expected in zip(aopc_output.desc, curr_expected_desc):
            assert_tensor_almost_equal(
                output.float(), expected.float(), delta=runtime_config.delta
            )
        for output, expected in zip(aopc_output.asc, curr_expected_asc):
            assert_tensor_almost_equal(
                output.float(), expected.float(), delta=runtime_config.delta
            )
        for output, expected in zip(aopc_output.rand, curr_expected_rand):
            assert_tensor_almost_equal(
                output.float(), expected.float(), delta=runtime_config.delta
            )

        aopcs_desc_list.append(aopc_output.desc)
        aopcs_asc_list.append(aopc_output.asc)
        aopcs_rand_list.append(aopc_output.rand)
    assert_all_tensors_almost_equal(aopcs_desc_list)
    assert_all_tensors_almost_equal(aopcs_asc_list)
    assert_all_tensors_almost_equal(aopcs_rand_list)
