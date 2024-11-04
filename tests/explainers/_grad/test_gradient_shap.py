import pytest  # noqa
import torch

from tests.explainers.utils import (
    make_config_for_explainer_with_grad_batch_size,
    run_explainer_test_with_config,
)

test_configurations = [
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_single_input_config",
        explainer="gradient_shap",
        expected=(torch.tensor([1.6415]), torch.tensor([-0.5804])),
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_single_input_config",
        explainer="gradient_shap",
        expected=None,
        override_target=0,
        throws_exception=True,
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_single_batched_input_config",
        explainer="gradient_shap",
        expected=(torch.tensor([[1.6415]]), torch.tensor([[-0.5804]])),
        override_target=0,
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_batch_input_config",
        explainer="gradient_shap",
        expected=(
            torch.tensor([1.4274, 1.2774, 1.2187]),
            torch.tensor([-0.4872, -0.0045, -0.0538]),
        ),
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        explainer="gradient_shap",
        expected=(
            torch.tensor([[0, 0, 0]]),
            torch.tensor([[0, 0, 0]]),
        ),
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        explainer="gradient_shap",
        expected=None,
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        explainer="gradient_shap",
        expected=[
            (
                torch.tensor(
                    [
                        [-29.7539, 74.0144, 96.4433],
                        [112.4857, 165.0676, 149.6760],
                        [225.7662, 275.6015, 263.9995],
                        [317.6607, 396.4174, 392.5694],
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [-3.7192, 9.2518, 12.0554],
                        [112.4857, 165.0676, 149.6760],
                        [225.7662, 275.6015, 263.9995],
                        [317.6607, 396.4174, 392.5694],
                    ]
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
        explainer="gradient_shap",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            -0.0051,
                            0.0499,
                            -0.0010,
                            0.0051,
                            0.0222,
                            -0.0660,
                            0.0043,
                            -0.0198,
                            0.0027,
                            -0.0976,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0007,
                            -0.0108,
                            -0.0013,
                            -0.0152,
                            -0.0188,
                            0.0012,
                            -0.0041,
                            -0.0115,
                            -0.0054,
                            0.0627,
                        ]
                    ]
                ),
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_softmax_model_single_input_single_target_config",
        explainer="gradient_shap",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            -5.5244e-04,
                            -8.8926e-04,
                            -4.0088e-03,
                            1.8327e-03,
                            1.3880e-03,
                            5.2164e-04,
                            -3.1749e-03,
                            -2.1243e-03,
                            1.6845e-05,
                            1.4502e-03,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0030,
                            -0.0021,
                            0.0028,
                            0.0003,
                            0.0018,
                            0.0027,
                            0.0023,
                            -0.0001,
                            -0.0006,
                            0.0019,
                        ]
                    ]
                ),
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_alexnet_model_config",
        explainer="gradient_shap",
        override_target=[0, 1, 2],
        expected=[None] * 3,
    ),
    *make_config_for_explainer_with_grad_batch_size(
        target_fixture="classification_alexnet_model_config",
        explainer="gradient_shap",
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
def test_gradient_shap(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration

    # gradient shap always requires a random baseline
    if isinstance(base_config.inputs, tuple):
        base_config.baselines = tuple(
            torch.randn((20, *x.shape[1:])) for x in base_config.inputs
        )
    else:
        base_config.baselines = torch.randn((20, *(base_config.inputs.shape[1:])))

    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
