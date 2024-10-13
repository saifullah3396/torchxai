import dataclasses

import pytest
import torch  # noqa

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
        explainer="kernel_shap",
        config_class=ExplainersTestRuntimeConfig_,
        internal_batch_sizes=[1, 20, 100],  # perturbation_batch_size
    )


test_configurations = [
    *_make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        expected=(torch.tensor([1.4898]), torch.tensor([-0.4898])),
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        expected=(
            torch.tensor([1.4898]),
            torch.tensor([-0.4898]),
        ),
        override_target=0,
        throws_exception=True,
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_single_batched_input_config",
        expected=(torch.tensor([[1.4898]]), torch.tensor([[-0.4898]])),
        override_target=0,
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_batch_input_config",
        expected=(
            torch.tensor([1.4898, 1.5510, 1.3980]),
            torch.tensor([-0.4898, -0.5510, -0.3980]),
        ),
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=(
            torch.tensor([[0.2364, 0.0028, -0.0453]]),
            torch.tensor([[-0.2154, -0.0592, 0.0807]]),
        ),
    ),
    *_make_config_for_explainer(
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=None,
    ),
    *_make_config_for_explainer(
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=[
            torch.tensor(
                [
                    [27.5679, 54.6671, 77.7651],
                    [127.9998, 160.0000, 192.0000],
                    [223.9999, 256.0000, 288.0000],
                    [320.0004, 352.0001, 384.0001],
                ]
            ),
            torch.tensor(
                [
                    [3.4460, 6.8334, 9.7206],
                    [127.9998, 160.0000, 192.0000],
                    [223.9999, 256.0000, 288.0000],
                    [320.0004, 352.0001, 384.0001],
                ]
            ),
        ],
        override_target=[
            [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
            [(0, 0, 0), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
        ],
        delta=1e-2,
    ),
    # *_make_config_for
    *_make_config_for_explainer(
        target_fixture="classification_sigmoid_model_single_input_single_target_config",
        expected=[
            (
                torch.tensor(
                    [
                        [
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            0.0000,
                            -0.0078,
                        ]
                    ]
                ),
            ),
            (torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    *_make_config_for_explainer(
        target_fixture="classification_softmax_model_multi_tuple_input_single_target_config",
        expected=[
            torch.tensor(
                [
                    [
                        -0.0058,
                        -0.0089,
                        -0.0091,
                        -0.0026,
                        0.0094,
                        0.0108,
                        -0.0012,
                        -0.0056,
                        -0.0197,
                        -0.0035,
                    ]
                ]
                * 3,
            ),
            torch.tensor([[0] * 10] * 3),
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
def test_kernel_shap(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration

    if runtime_config.set_image_feature_mask:
        base_config.feature_mask = grid_segmenter(base_config.inputs, cell_size=32)

    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
