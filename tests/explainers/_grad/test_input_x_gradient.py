import pytest  # noqa
import torch

from tests.explainers.utils import (
    make_config_for_explainer,
    run_explainer_test_with_config,
)

test_configurations = [
    make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        explainer="input_x_gradient",
        expected=(torch.tensor([3.0]), torch.tensor([-1.0])),
    ),
    make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        explainer="input_x_gradient",
        expected=(torch.tensor([3.0]), torch.tensor([-1.0])),
        override_target=0,
        throws_exception=True,
    ),
    make_config_for_explainer(
        target_fixture="basic_model_single_batched_input_config",
        explainer="input_x_gradient",
        expected=(torch.tensor([[3.0]]), torch.tensor([[-1.0]])),
        override_target=0,
    ),
    make_config_for_explainer(
        target_fixture="basic_model_batch_input_config",
        explainer="input_x_gradient",
        expected=(
            torch.tensor([3, 3, 3]),
            torch.tensor([-1, -1, -1]),
        ),
    ),
    make_config_for_explainer(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        explainer="input_x_gradient",
        expected=(
            torch.tensor([[0, 0, 0]]),
            torch.tensor([[0, 0, 0]]),
        ),
    ),
    make_config_for_explainer(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="input_x_gradient",
        expected=(
            torch.tensor([[1, 4, 6, 4], [5, 12, 14, 8], [9, 20, 22, 12], [0, 0, 0, 0]])
            .unsqueeze(0)
            .repeat(20, 1, 1, 1)
        ),
    ),
    make_config_for_explainer(
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        explainer="input_x_gradient",
        expected=[
            torch.tensor(
                [
                    [32.0, 64.0, 96.0],
                    [128.0, 160.0, 192.0],
                    [224.0, 256.0, 288.0],
                    [320.0, 352.0, 384.0],
                ]
            ),
            torch.tensor(
                [
                    [4.0, 8.0, 12.0],
                    [128.0, 160.0, 192.0],
                    [224.0, 256.0, 288.0],
                    [320.0, 352.0, 384.0],
                ]
            ),
        ],
        override_target=[
            [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
            [(0, 0, 0), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
        ],
    ),
    make_config_for_explainer(
        target_fixture="classification_sigmoid_model_single_input_single_target_config",
        explainer="input_x_gradient",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            0.0157,
                            0.0233,
                            0.0016,
                            -0.0218,
                            0.0163,
                            -0.0256,
                            0.0075,
                            -0.0041,
                            -0.0143,
                            -0.0398,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            -0.0029,
                            0.0065,
                            -0.0146,
                            -0.0205,
                            -0.0104,
                            0.0038,
                            0.0160,
                            -0.0005,
                            -0.0218,
                            0.0233,
                        ]
                    ]
                ),
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    make_config_for_explainer(
        target_fixture="classification_softmax_model_single_input_single_target_config",
        explainer="input_x_gradient",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            -0.0025,
                            -0.0002,
                            -0.0024,
                            0.0006,
                            0.0010,
                            -0.0005,
                            0.0007,
                            0.0007,
                            0.0014,
                            -0.0027,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0047,
                            0.0001,
                            0.0021,
                            0.0016,
                            0.0024,
                            0.0064,
                            -0.0019,
                            -0.0027,
                            -0.0040,
                            0.0036,
                        ]
                    ]
                ),
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    make_config_for_explainer(
        target_fixture="classification_alexnet_model_config",
        explainer="input_x_gradient",
        override_target=[0, 1, 2],
        expected=[None] * 3,
    ),
    make_config_for_explainer(
        target_fixture="classification_alexnet_model_config",
        explainer="input_x_gradient",
        override_target=[
            [0] * 10,
            [1] * 10,
            list(range(10)),
        ],  # take all the outputs at 0th index as target
        expected=[None] * 3,
    ),
]


@pytest.mark.explainers
@pytest.mark.parametrize(
    "explainer_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_input_x_gradient(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration
    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
