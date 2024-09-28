import dataclasses

import pytest  # noqa
import torch

from tests.explainers.utils import (
    ExplainersTestRuntimeConfig,
    make_config_for_explainers_with_internal_batch_size,
    run_explainer_test_with_config,
)
from tests.utils.common import grid_segmenter


@dataclasses.dataclass
class ExplainersTestRuntimeConfig_(ExplainersTestRuntimeConfig):
    set_image_feature_mask: bool = False


def _make_config_for_explainer(
    *args,
    **kwargs,
):
    return make_config_for_explainers_with_internal_batch_size(
        *args,
        **kwargs,
        explainer="feature_ablation",
        config_class=ExplainersTestRuntimeConfig_,
        internal_batch_sizes=[1, 20, 100],  # perturbation_batch_size
    )


test_configurations = [
    *_make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        expected=(
            torch.tensor([1.0]),
            torch.tensor([-1.0]),
        ),
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        expected=(
            torch.tensor([1.0]),
            torch.tensor([-1.0]),
        ),
        override_target=0,
        throws_exception=True,
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_single_batched_input_config",
        expected=(torch.tensor([[1.0]]), torch.tensor([[-1.0]])),
        override_target=0,
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_batch_input_config",
        expected=(
            torch.tensor([1, 1, 1]),
            torch.tensor([-1, -1, -1]),
        ),
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=(
            torch.tensor([[0, 0, 0]]),
            torch.tensor([[-0.5, 0, 0]]),
        ),
    ),
    *_make_config_for_explainer(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=(
            torch.tensor(
                [
                    [1.0, 4.0, 6.0, 4.0],
                    [4.0, 10.0, 11.0, 8.0],
                    [4.0, 14.0, 15.0, 12.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
            .unsqueeze(0)
            .repeat(20, 1, 1, 1)
        ),
    ),
    *_make_config_for_explainer(
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=[
            (
                torch.tensor(
                    [
                        [32.0, 64.0, 88.0],
                        [128.0, 160.0, 192.0],
                        [224.0, 256.0, 288.0],
                        [320.0, 352.0, 384.0],
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [4.0, 8.0, 11.0],
                        [128.0, 160.0, 192.0],
                        [224.0, 256.0, 288.0],
                        [320.0, 352.0, 384.0],
                    ]
                ),
            ),
        ],
        override_target=[
            [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
            [(0, 0, 0), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
        ],
    ),
    *_make_config_for_explainer(
        target_fixture="classification_sigmoid_model_single_input_single_target_config",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            0.0049,
                            0.0232,
                            0.0067,
                            -0.0235,
                            0.0079,
                            -0.0225,
                            0.0075,
                            -0.0078,
                            -0.0139,
                            -0.0445,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0093,
                            0.0065,
                            -0.0088,
                            -0.0175,
                            -0.0015,
                            0.0075,
                            0.0159,
                            0.0035,
                            -0.0197,
                            0.0292,
                        ]
                    ]
                ),
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    *_make_config_for_explainer(
        target_fixture="classification_softmax_model_single_input_single_target_config",
        expected=[
            torch.tensor(
                [
                    [
                        -7.0127e-04,
                        4.7706e-04,
                        -1.5219e-03,
                        8.3491e-05,
                        -4.4073e-04,
                        -8.4874e-04,
                        6.9338e-04,
                        8.9946e-04,
                        1.8473e-03,
                        -2.4041e-04,
                    ]
                ],
            ),
            torch.tensor(
                [
                    [
                        5.0232e-03,
                        1.1835e-04,
                        1.5855e-03,
                        4.4607e-05,
                        1.4812e-03,
                        5.2779e-03,
                        -1.8666e-03,
                        -2.3631e-03,
                        -5.4629e-03,
                        2.5512e-03,
                    ]
                ]
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    *_make_config_for_explainer(
        target_fixture="classification_alexnet_model_config",
        override_target=[0, 1, 2],
        expected=[None] * 3,
        set_image_feature_mask=True,
    ),
    *_make_config_for_explainer(
        target_fixture="classification_alexnet_model_config",
        override_target=[
            [0] * 10,
            [1] * 10,
            list(range(10)),
        ],  # take all the outputs at 0th index as target
        expected=[None] * 3,
        set_image_feature_mask=True,
    ),
]


@pytest.mark.explainers
@pytest.mark.parametrize(
    "explainer_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_feature_ablation(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration

    if runtime_config.set_image_feature_mask:
        base_config.feature_mask = grid_segmenter(base_config.inputs, cell_size=32)

    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
