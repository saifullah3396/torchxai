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
            torch.tensor(
                [
                    [
                        0.0103,
                        0.0172,
                        0.0015,
                        -0.0075,
                        0.0151,
                        -0.0255,
                        0.0054,
                        -0.0079,
                        -0.0029,
                        -0.0353,
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        0.0016,
                        -0.0094,
                        -0.0061,
                        -0.0157,
                        -0.0110,
                        0.0095,
                        0.0145,
                        -0.0037,
                        -0.0229,
                        0.0248,
                    ]
                ]
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
    ),
    *_make_config_for_explainer(
        target_fixture="classification_softmax_model_multi_tuple_input_single_target_config",
        expected=[
            torch.tensor(
                [
                    [
                        -2.1904e-04,
                        -5.0956e-04,
                        -1.5240e-03,
                        3.6477e-04,
                        1.6572e-03,
                        2.0785e-03,
                        4.5674e-05,
                        -1.8594e-04,
                        -2.1296e-03,
                        -7.7510e-04,
                    ],
                    [
                        3.0074e-05,
                        -8.5715e-04,
                        -8.3074e-04,
                        1.7218e-04,
                        1.7893e-03,
                        1.4594e-03,
                        -3.9504e-05,
                        -1.4519e-05,
                        -2.1594e-03,
                        -1.0053e-03,
                    ],
                    [
                        8.2257e-04,
                        -2.0941e-03,
                        -8.9770e-04,
                        -2.4427e-04,
                        1.9496e-03,
                        1.4402e-03,
                        4.4668e-04,
                        -5.6682e-04,
                        -1.8416e-03,
                        -2.3448e-04,
                    ],
                ]
            ),
            torch.tensor(
                [
                    [
                        1.3142e-03,
                        -5.3121e-04,
                        2.9035e-03,
                        7.9721e-05,
                        1.2539e-03,
                        9.0579e-04,
                        -5.9787e-04,
                        -6.8029e-04,
                        1.2287e-03,
                        -6.7752e-04,
                    ],
                    [
                        1.3477e-03,
                        -3.6288e-04,
                        2.7747e-03,
                        -1.8235e-04,
                        1.4236e-03,
                        1.0422e-03,
                        -3.5693e-04,
                        -1.1297e-03,
                        9.6906e-04,
                        -6.8290e-04,
                    ],
                    [
                        7.1941e-04,
                        1.6579e-04,
                        1.8111e-03,
                        1.6382e-04,
                        2.2750e-03,
                        6.7806e-04,
                        -2.4891e-05,
                        -3.9932e-04,
                        1.4099e-03,
                        -1.9354e-03,
                    ],
                ]
            ),
        ],
        override_target=[torch.tensor([0]), torch.tensor([1])],
        delta=1e-3,
    ),
    *_make_config_for_explainer(
        target_fixture="classification_alexnet_model_config_single_sample",
        override_target=[0, 1, 2],
        expected=[None] * 3,
        set_image_feature_mask=True,
        visualize=False,
    ),
    *_make_config_for_explainer(
        target_fixture="classification_alexnet_model_real_images_single_sample_config",
        override_target=[0, 1, 2],
        expected=[None] * 3,
        set_image_feature_mask=True,
        visualize=False,
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
