import pytest
import torch  # noqa

from tests.explainers.utils import (
    make_config_for_explainer,
    run_explainer_test_with_config,
)


def _make_config_for_explainer(
    *args,
    **kwargs,
):
    strides = kwargs.pop("strides", None)
    sliding_window_shapes = kwargs.pop("sliding_window_shapes", None)
    return [
        make_config_for_explainer(
            *args,
            **kwargs,
            explainer_kwargs={
                "internal_batch_size": internal_batch_size,
                "strides": strides,
                "sliding_window_shapes": sliding_window_shapes,
            },
            explainer="occlusion",
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in [1, 20, 100]
    ]


test_configurations = [
    *_make_config_for_explainer(
        target_fixture="classification_alexnet_model_config",
        override_target=[0, 1, 2],
        expected=[None] * 3,
        strides=(3, 16, 16),
        sliding_window_shapes=(3, 32, 32),
        check_multi_target_list_against_single_target=True,
    ),
    *_make_config_for_explainer(
        target_fixture="classification_alexnet_model_config",
        override_target=[
            [0] * 10,
            [1] * 10,
            list(range(10)),
        ],  # take all the outputs at 0th index as target
        expected=[None] * 3,
        strides=(3, 16, 16),
        sliding_window_shapes=(3, 32, 32),
        check_multi_target_list_against_single_target=True,
    ),
]


@pytest.mark.explainers
@pytest.mark.parametrize(
    "explainer_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_occlusion(explainer_runtime_test_configuration):
    base_config, runtime_config = explainer_runtime_test_configuration

    run_explainer_test_with_config(
        base_config=base_config, runtime_config=runtime_config
    )
