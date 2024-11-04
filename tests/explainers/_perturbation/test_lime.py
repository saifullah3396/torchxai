import dataclasses

import pytest
import torch  # noqa

from tests.explainers.utils import (
    ExplainersTestRuntimeConfig,
    make_config_for_explainer_with_internal_batch_size,
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
    return make_config_for_explainer_with_internal_batch_size(
        *args,
        **kwargs,
        explainer="lime",
        config_class=ExplainersTestRuntimeConfig_,
        internal_batch_sizes=[1, 20, 100],  # perturbation_batch_size
    )


test_configurations = [
    *_make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        expected=(torch.tensor([1.3842]), torch.tensor([-0.5370])),
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        expected=(
            torch.tensor([1.3842]),
            torch.tensor([-0.5370]),
        ),
        override_target=0,
        throws_exception=True,
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_single_batched_input_config",
        expected=(torch.tensor([[1.3842]]), torch.tensor([[-0.5370]])),
        override_target=0,
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_batch_input_config",
        expected=(
            torch.tensor([1.3842, 1.4314, 1.4280]),
            torch.tensor([-0.5370, -0.4566, -0.5858]),
        ),
    ),
    *_make_config_for_explainer(
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=(
            torch.tensor([[0.2236, 0.0000, 0.0000]]),
            torch.tensor([[-0.1866, 0, 0]]),
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
                    [26.9094, 54.5994, 79.0141],
                    [127.9639, 159.9558, 191.9592],
                    [223.9636, 255.9571, 287.9532],
                    [319.9602, 351.9643, 383.9579],
                ]
            ),
            torch.tensor(
                [
                    [3.3217, 6.7899, 9.8324],
                    [127.9626, 159.9561, 191.9605],
                    [223.9644, 255.9591, 287.9551],
                    [319.9585, 351.9663, 383.9557],
                ]
            ),
        ],
        override_target=[
            [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
            [(0, 0, 0), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
        ],
        delta=1e-2,
    ),
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
                [[0] * 10] * 3,
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
def test_lime(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration

    if runtime_config.set_image_feature_mask:
        base_config.feature_mask = grid_segmenter(base_config.inputs, cell_size=32)

    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
