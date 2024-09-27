import pytest  # noqa
import torch

from tests.explainers.utils import (
    make_config_for_explainer,
    run_explainer_test_with_config,
)

test_configurations = [
    make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        explainer="deep_lift",
        expected=(torch.tensor([3.0]), torch.tensor([-1.0])),
    ),
    make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        explainer="deep_lift",
        expected=(torch.tensor([3.0]), torch.tensor([-1.0])),
        override_target=0,
        throws_exception=True,
    ),
    make_config_for_explainer(
        target_fixture="basic_model_single_batched_input_config",
        explainer="deep_lift",
        expected=(torch.tensor([[3.0]]), torch.tensor([[-1.0]])),
        override_target=0,
    ),
    make_config_for_explainer(
        target_fixture="basic_model_batch_input_config",
        explainer="deep_lift",
        expected=(
            torch.tensor([3, 3, 3]),
            torch.tensor([-1, -1, -1]),
        ),
    ),
    make_config_for_explainer(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        explainer="deep_lift",
        expected=(
            torch.tensor([[0, 0, 0]]),
            torch.tensor([[0, 0, 0]]),
        ),
    ),
    make_config_for_explainer(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="deep_lift",
        expected=(
            torch.tensor(
                [
                    [
                        [0.0741, 0.5608, 0.8413, 0.8254],
                        [1.7593, 4.8644, 5.6751, 3.6710],
                        [3.1667, 8.1073, 8.9180, 5.5065],
                        [3.6111, 7.4242, 7.9545, 4.0404],
                    ]
                ]
            )
            .unsqueeze(0)
            .repeat(20, 1, 1, 1)
        ),
    ),
    make_config_for_explainer(
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        explainer="deep_lift",
        expected=[
            (
                torch.tensor(
                    [
                        [26.6667, 53.3333, 80.0000],
                        [128.0000, 160.0000, 192.0000],
                        [224.0000, 256.0000, 288.0000],
                        [320.0000, 352.0000, 384.0000],
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [3.3333, 6.6667, 10.0000],
                        [128.0000, 160.0000, 192.0000],
                        [224.0000, 256.0000, 288.0000],
                        [320.0000, 352.0000, 384.0000],
                    ]
                ),
            ),
        ],
        override_target=[
            [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
            [(0, 0, 0), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
        ],
    ),
    make_config_for_explainer(
        target_fixture="classification_sigmoid_model_single_input_single_target_config",
        explainer="deep_lift",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            0.0145,
                            0.0263,
                            -0.0018,
                            -0.0162,
                            0.0174,
                            -0.0246,
                            0.0083,
                            -0.0043,
                            -0.0087,
                            -0.0404,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            -0.0034,
                            0.0028,
                            -0.0160,
                            -0.0191,
                            -0.0055,
                            0.0081,
                            0.0127,
                            -0.0014,
                            -0.0182,
                            0.0217,
                        ]
                    ],
                ),
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    make_config_for_explainer(
        target_fixture="classification_softmax_model_single_input_single_target_config",
        explainer="deep_lift",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            -2.7941e-03,
                            -1.2082e-03,
                            -1.4229e-03,
                            1.2388e-03,
                            -6.4839e-04,
                            -4.9929e-04,
                            -1.0269e-03,
                            -5.0402e-04,
                            1.7938e-03,
                            3.4616e-05,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0022,
                            -0.0007,
                            0.0013,
                            0.0015,
                            0.0011,
                            0.0050,
                            -0.0005,
                            -0.0019,
                            -0.0013,
                            0.0020,
                        ]
                    ]
                ),
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    make_config_for_explainer(
        target_fixture="classification_alexnet_model_config",
        explainer="deep_lift",
        override_target=[0, 1, 2],
        expected=[None] * 3,
    ),
    make_config_for_explainer(
        target_fixture="classification_alexnet_model_config",
        explainer="deep_lift",
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
def test_deep_lift(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration
    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
