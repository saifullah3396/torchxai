import pytest  # noqa
import torch

from tests.explainers.utils import (
    make_config_for_explainer,
    run_explainer_test_with_config,
)

test_configurations = [
    *[
        make_config_for_explainer(
            target_fixture="basic_model_single_input_config",
            explainer="integrated_gradients",
            expected=(torch.tensor([1.5]), torch.tensor([-0.5])),
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [None, 1, 4]
    ],
    make_config_for_explainer(
        target_fixture="basic_model_single_input_config",
        explainer="integrated_gradients",
        expected=(torch.tensor([1.5]), torch.tensor([-0.5])),
        override_target=0,
        throws_exception=True,
    ),
    *[
        make_config_for_explainer(
            target_fixture="basic_model_single_batched_input_config",
            explainer="integrated_gradients",
            expected=(torch.tensor([[1.5]]), torch.tensor([[-0.5]])),
            override_target=0,
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [None, 1, 4]
    ],
    *[
        make_config_for_explainer(
            target_fixture="basic_model_batch_input_config",
            explainer="integrated_gradients",
            expected=(
                torch.tensor([1.5, 1.5, 1.5]),
                torch.tensor([-0.5, -0.5, -0.5]),
            ),
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [None, 1, 4]
    ],
    *[
        make_config_for_explainer(
            target_fixture="basic_model_batch_input_with_additional_forward_args_config",
            explainer="integrated_gradients",
            expected=(
                torch.tensor([[0, 0, 0]]),
                torch.tensor([[0, 0, 0]]),
            ),
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [None, 1, 4]
    ],
    *[
        make_config_for_explainer(
            target_fixture="classification_convnet_model_with_multiple_targets_config",
            explainer="integrated_gradients",
            expected=torch.tensor(
                [
                    [0.0805, 0.5781, 0.8672, 0.8344],
                    [1.7900, 4.9930, 5.8252, 3.7934],
                    [3.2220, 8.3217, 9.1539, 5.6901],
                    [3.6078, 7.6034, 8.1465, 4.2492],
                ]
            )
            .unsqueeze(0)
            .repeat(20, 1, 1, 1),
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [None, 1, 4]
    ],
    *[
        make_config_for_explainer(
            target_fixture="classification_multilayer_model_with_tuple_targets_config",
            explainer="integrated_gradients",
            expected=[
                (
                    torch.tensor(
                        [
                            [26.7756, 53.5513, 80.3269],
                            [128.0000, 160.0000, 192.0000],
                            [224.0000, 256.0000, 288.0000],
                            [320.0000, 352.0000, 384.0000],
                        ]
                    ),
                ),
                (
                    torch.tensor(
                        [
                            [3.3470, 6.6939, 10.0409],
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
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [None, 1, 4]
    ],
    *[
        make_config_for_explainer(
            target_fixture="classification_sigmoid_model_single_input_single_target_config",
            explainer="integrated_gradients",
            expected=[
                (
                    torch.tensor(
                        [
                            [
                                0.0145,
                                0.0262,
                                -0.0017,
                                -0.0164,
                                0.0174,
                                -0.0246,
                                0.0083,
                                -0.0044,
                                -0.0088,
                                -0.0403,
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
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [None, 1, 4]
    ],
    *[
        make_config_for_explainer(
            target_fixture="classification_softmax_model_single_input_single_target_config",
            explainer="integrated_gradients",
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
                                0.0044,
                                0.0015,
                                0.0008,
                                -0.0008,
                                0.0009,
                                0.0041,
                                0.0002,
                                -0.0014,
                                -0.0042,
                                0.0031,
                            ]
                        ]
                    ),
                ),
            ],
            override_target=[torch.tensor([0]), torch.tensor([1])],
            delta=1e-3,
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [None, 1, 4]
    ],
    *[
        make_config_for_explainer(
            target_fixture="classification_alexnet_model_config",
            explainer="integrated_gradients",
            override_target=[0, 1, 2],
            expected=[None] * 3,
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [64, 1, 4]
    ],
    *[
        make_config_for_explainer(
            target_fixture="classification_alexnet_model_config",
            explainer="integrated_gradients",
            override_target=[
                [0] * 10,
                [1] * 10,
                list(range(10)),
            ],  # take all the outputs at 0th index as target
            expected=[None] * 3,
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [64, 1, 4]
    ],
]


@pytest.mark.explainers
@pytest.mark.parametrize(
    "explainer_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_integrated_gradients(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration
    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
