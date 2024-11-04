import pytest  # noqa
import torch

from tests.explainers.utils import (
    make_config_for_explainer_with_grad_batch_size,
    run_explainer_test_with_config,
)

test_configurations = [
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_single_input_config",
        explainer="guided_backprop",
        expected=(torch.tensor([1.0]), torch.tensor([-1.0])),
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_single_input_config",
        explainer="guided_backprop",
        expected=(torch.tensor([1.0]), torch.tensor([-1.0])),
        override_target=0,
        throws_exception=True,
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_single_batched_input_config",
        explainer="guided_backprop",
        expected=(torch.tensor([[1.0]]), torch.tensor([[-1.0]])),
        override_target=0,
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_batch_input_config",
        explainer="guided_backprop",
        expected=(
            torch.tensor([1, 1, 1]),
            torch.tensor([-1, -1, -1]),
        ),
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        explainer="guided_backprop",
        expected=(
            torch.tensor([[0, 0, 0]]),
            torch.tensor([[0, 0, 0]]),
        ),
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="guided_backprop",
        expected=(
            torch.tensor([[1, 2, 2, 1], [2, 4, 4, 2], [2, 4, 4, 2], [1, 2, 2, 1]])
            .unsqueeze(0)
            .repeat(20, 1, 1, 1)
        ),
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        explainer="guided_backprop",
        expected=[
            (torch.tensor([32]).expand(4, 3),),
            (
                torch.cat(
                    [torch.tensor(4).expand(1, 3), torch.tensor([32]).expand(3, 3)],
                    dim=0,
                ),
            ),
        ],
        override_target=[
            [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
            [(0, 0, 0), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
        ],
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_sigmoid_model_single_input_single_target_config",
        explainer="guided_backprop",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            0.0192,
                            0.0109,
                            0.0129,
                            -0.0030,
                            0.0182,
                            0.0138,
                            0.0028,
                            -0.0147,
                            0.0061,
                            -0.0125,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0141,
                            0.0017,
                            0.0027,
                            -0.0140,
                            0.0056,
                            0.0257,
                            0.0139,
                            -0.0090,
                            -0.0061,
                            0.0107,
                        ]
                    ]
                ),
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_softmax_model_single_input_single_target_config",
        explainer="guided_backprop",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            3.7760e-04,
                            8.5276e-05,
                            2.0446e-04,
                            1.6386e-04,
                            1.1548e-03,
                            1.5595e-03,
                            1.1076e-04,
                            -5.6630e-04,
                            1.8906e-04,
                            -2.2819e-05,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0046,
                            0.0001,
                            0.0025,
                            0.0015,
                            0.0028,
                            0.0061,
                            -0.0013,
                            -0.0024,
                            -0.0027,
                            0.0025,
                        ]
                    ]
                ),
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_alexnet_model_config",
        explainer="guided_backprop",
        override_target=[0, 1, 2],
        expected=[None] * 3,
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_alexnet_model_config",
        explainer="guided_backprop",
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
def test_guided_backprop(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration
    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
