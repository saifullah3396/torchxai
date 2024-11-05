import dataclasses

import pytest  # noqa

from tests.utils.common import (
    assert_tensor_almost_equal,
    grid_segmenter,
    set_all_random_seeds,
)
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics import monotonicity


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    max_features_processed_per_batch: int = None
    set_image_feature_mask: bool = True


test_configurations = [
    MetricTestRuntimeConfig_(
        test_name="classification_alexnet_model",
        target_fixture="classification_alexnet_model_config",
        explainer="saliency",
        override_target=[0, 1, 2],
        explainer_kwargs={"is_multi_target": True},
        delta=1e-8,
        max_features_processed_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_alexnet_model",
        target_fixture="classification_alexnet_model_config",
        explainer="integrated_gradients",
        override_target=[0, 1, 2],
        explainer_kwargs={"is_multi_target": True},
        delta=1e-8,
        max_features_processed_per_batch=[5, 1, 40],
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_monotonicity_multi_target(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration

    assert len(explanations) == len(
        runtime_config.override_target
    ), "Number of explanations should be equal to the number of targets"

    if runtime_config.set_image_feature_mask:
        base_config.feature_mask = grid_segmenter(base_config.inputs, cell_size=32)
        base_config.feature_mask = base_config.feature_mask.expand_as(
            base_config.inputs
        )

    runtime_config.max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )

    for max_features in runtime_config.max_features_processed_per_batch:
        set_all_random_seeds(1234)
        monotonicity_batch_list_1, fwd_features_batch_list_1 = monotonicity(
            forward_func=base_config.model,
            inputs=base_config.inputs,
            attributions=explanations,
            baselines=base_config.baselines,
            feature_mask=base_config.feature_mask,
            additional_forward_args=base_config.additional_forward_args,
            target=base_config.target,
            max_features_processed_per_batch=max_features,
            is_multi_target=True,
            return_intermediate_results=True,
        )

        set_all_random_seeds(1234)
        monotonicity_batch_list_2 = []
        fwd_features_batch_list_2 = []
        for explanation, target in zip(explanations, runtime_config.override_target):
            monotonicity_batch, fwd_features_batch = monotonicity(
                forward_func=base_config.model,
                inputs=base_config.inputs,
                attributions=explanation,
                baselines=base_config.baselines,
                feature_mask=base_config.feature_mask,
                additional_forward_args=base_config.additional_forward_args,
                target=target,
                max_features_processed_per_batch=max_features,
                return_intermediate_results=True,
            )
            monotonicity_batch_list_2.append(monotonicity_batch)
            fwd_features_batch_list_2.append(fwd_features_batch)

        assert len(monotonicity_batch_list_1) == len(monotonicity_batch_list_2)
        assert len(fwd_features_batch_list_1) == len(fwd_features_batch_list_2)

        for x, y in zip(monotonicity_batch_list_1, monotonicity_batch_list_2):
            for xx, yy in zip(x, y):
                assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
        for x, y in zip(fwd_features_batch_list_1, fwd_features_batch_list_2):
            for xx, yy in zip(x, y):
                assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
