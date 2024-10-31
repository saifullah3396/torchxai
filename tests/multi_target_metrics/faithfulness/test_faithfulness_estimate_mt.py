import dataclasses

import pytest  # noqa

from tests.utils.common import (
    assert_tensor_almost_equal,
    grid_segmenter,
    set_all_random_seeds,
)
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics import faithfulness_estimate


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
def test_non_sensitivity(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration

    assert len(explanations) == len(
        runtime_config.override_target
    ), "Number of explanations should be equal to the number of targets"

    if runtime_config.set_image_feature_mask:
        base_config.feature_mask = grid_segmenter(base_config.inputs, cell_size=32)

    runtime_config.max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )

    for max_features in runtime_config.max_features_processed_per_batch:
        set_all_random_seeds(1234)
        (
            faithfulness_estimate_batch_list_1,
            attributions_sum_perturbed_batch_list_1,
            inputs_perturbed_fwd_diffs_batch_list_1,
        ) = faithfulness_estimate(
            forward_func=base_config.model,
            inputs=base_config.inputs,
            attributions=[
                explanation.sum(dim=1, keepdim=True) for explanation in explanations
            ],
            baselines=base_config.baselines,
            feature_mask=base_config.feature_mask,
            additional_forward_args=base_config.additional_forward_args,
            target=base_config.target,
            max_features_processed_per_batch=max_features,
            is_multi_target=True,
        )

        set_all_random_seeds(1234)
        faithfulness_estimate_batch_list_2 = []
        attributions_sum_perturbed_batch_list_2 = []
        inputs_perturbed_fwd_diffs_batch_list_2 = []
        for explanation, target in zip(explanations, runtime_config.override_target):
            (
                faithfulness_estimate_batch,
                attributions_sum_perturbed_batch,
                inputs_perturbed_fwd_diffs_batch,
            ) = faithfulness_estimate(
                forward_func=base_config.model,
                inputs=base_config.inputs,
                attributions=explanation.sum(dim=1, keepdim=True),
                baselines=base_config.baselines,
                feature_mask=base_config.feature_mask,
                additional_forward_args=base_config.additional_forward_args,
                target=target,
                max_features_processed_per_batch=max_features,
            )
            faithfulness_estimate_batch_list_2.append(faithfulness_estimate_batch)
            attributions_sum_perturbed_batch_list_2.append(
                attributions_sum_perturbed_batch
            )
            inputs_perturbed_fwd_diffs_batch_list_2.append(
                inputs_perturbed_fwd_diffs_batch
            )

        assert len(faithfulness_estimate_batch_list_1) == len(
            faithfulness_estimate_batch_list_2
        )
        assert len(attributions_sum_perturbed_batch_list_1) == len(
            attributions_sum_perturbed_batch_list_2
        )
        assert len(inputs_perturbed_fwd_diffs_batch_list_1) == len(
            inputs_perturbed_fwd_diffs_batch_list_2
        )

        for x, y in zip(
            faithfulness_estimate_batch_list_1, faithfulness_estimate_batch_list_2
        ):
            for xx, yy in zip(x, y):
                assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
        for x, y in zip(
            attributions_sum_perturbed_batch_list_1,
            attributions_sum_perturbed_batch_list_2,
        ):
            for xx, yy in zip(x, y):
                assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
        for x, y in zip(
            inputs_perturbed_fwd_diffs_batch_list_1,
            inputs_perturbed_fwd_diffs_batch_list_2,
        ):
            for xx, yy in zip(x, y):
                assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
