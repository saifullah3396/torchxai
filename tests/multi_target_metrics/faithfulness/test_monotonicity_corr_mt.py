import dataclasses
from typing import Callable

import pytest  # noqa

from tests.utils.common import (
    assert_tensor_almost_equal,
    grid_segmenter,
    set_all_random_seeds,
)
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics import monotonicity_corr_and_non_sens
from torchxai.metrics._utils.perturbation import default_random_perturb_func


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    perturb_func: Callable = default_random_perturb_func()
    n_perturbations_per_feature: int = 100
    max_features_processed_per_batch: int = None
    set_image_feature_mask: bool = False


test_configurations = [
    MetricTestRuntimeConfig_(
        test_name="classification_alexnet_model",
        target_fixture="classification_alexnet_model_single_sample_config",
        explainer="saliency",
        override_target=[0, 1, 2],
        expected=None,
        explainer_kwargs={"is_multi_target": True},
        delta=1e-8,
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[1, None, 40],
        set_image_feature_mask=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_alexnet_model",
        target_fixture="classification_alexnet_model_real_images_single_sample_config",
        explainer="saliency",
        override_target=[0, 1, 2],
        expected=None,
        explainer_kwargs={"is_multi_target": True},
        delta=1e-8,
        n_perturbations_per_feature=[10, 10, 20],
        max_features_processed_per_batch=[1, None, 40],
        set_image_feature_mask=True,
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_monotonicity_corr_multi_target(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration

    assert len(explanations) == len(
        runtime_config.override_target
    ), "Number of explanations should be equal to the number of targets"

    if runtime_config.set_image_feature_mask:
        base_config.feature_mask = grid_segmenter(base_config.inputs, cell_size=32)
        base_config.feature_mask = base_config.feature_mask.expand_as(
            base_config.inputs
        )

    runtime_config.n_perturbations_per_feature = _format_to_list(
        runtime_config.n_perturbations_per_feature
    )
    runtime_config.max_features_processed_per_batch = _format_to_list(
        runtime_config.max_features_processed_per_batch
    )
    runtime_config.expected = _format_to_list(runtime_config.expected)

    assert len(runtime_config.n_perturbations_per_feature) == len(
        runtime_config.max_features_processed_per_batch
    )
    assert (
        len(runtime_config.n_perturbations_per_feature) == len(runtime_config.expected)
        or len(runtime_config.expected) == 1
    )

    for n_perturbs, max_features in zip(
        runtime_config.n_perturbations_per_feature,
        runtime_config.max_features_processed_per_batch,
    ):
        set_all_random_seeds(1234)
        (
            monotonicity_corr_score_batch_list_1,
            _,
            _,
            perturbed_fwd_diffs_relative_vars_batch_list_1,
            feature_group_attribution_scores_batch_list_1,
        ) = monotonicity_corr_and_non_sens(
            forward_func=base_config.model,
            inputs=base_config.inputs,
            attributions=explanations,
            feature_mask=base_config.feature_mask,
            additional_forward_args=base_config.additional_forward_args,
            target=runtime_config.override_target,
            perturb_func=runtime_config.perturb_func,
            n_perturbations_per_feature=n_perturbs,
            max_features_processed_per_batch=max_features,
            show_progress=False,
            is_multi_target=True,
            return_intermediate_results=True,
            return_ratio=False,
        )
        monotonicity_corr_score_batch_list_2 = []
        perturbed_fwd_diffs_relative_vars_batch_list_2 = []
        feature_group_attribution_scores_batch_list_2 = []
        for explanation, target in zip(explanations, runtime_config.override_target):
            set_all_random_seeds(1234)
            # for this test we take the sum of the explanations over channel dimension to match the feature dimension
            # of the feature mask
            (
                monotonicity_corr_score_batch,
                _,
                _,
                perturbed_fwd_diffs_relative_vars_batch,
                feature_group_attribution_scores_batch,
            ) = monotonicity_corr_and_non_sens(
                forward_func=base_config.model,
                inputs=base_config.inputs,
                attributions=explanation,
                feature_mask=base_config.feature_mask,
                additional_forward_args=base_config.additional_forward_args,
                target=target,
                perturb_func=runtime_config.perturb_func,
                n_perturbations_per_feature=n_perturbs,
                max_features_processed_per_batch=max_features,
                show_progress=False,
                return_intermediate_results=True,
                return_ratio=False,
            )
            monotonicity_corr_score_batch_list_2.append(monotonicity_corr_score_batch)
            perturbed_fwd_diffs_relative_vars_batch_list_2.append(
                perturbed_fwd_diffs_relative_vars_batch
            )
            feature_group_attribution_scores_batch_list_2.append(
                feature_group_attribution_scores_batch
            )
        assert len(monotonicity_corr_score_batch_list_1) == len(
            monotonicity_corr_score_batch_list_2
        )
        assert len(perturbed_fwd_diffs_relative_vars_batch_list_1) == len(
            perturbed_fwd_diffs_relative_vars_batch_list_2
        )
        assert len(feature_group_attribution_scores_batch_list_1) == len(
            feature_group_attribution_scores_batch_list_2
        )
        import torch

        for x, y in zip(
            monotonicity_corr_score_batch_list_1, monotonicity_corr_score_batch_list_2
        ):
            assert_tensor_almost_equal(
                x.float(), y.float(), delta=runtime_config.delta, mode="mean"
            )
        for x, y in zip(
            perturbed_fwd_diffs_relative_vars_batch_list_1,
            perturbed_fwd_diffs_relative_vars_batch_list_2,
        ):
            for xx, yy in zip(x, y):
                xx = xx / torch.max(xx)
                yy = yy / torch.max(yy)
                assert_tensor_almost_equal(
                    xx.float(),
                    yy.float(),
                    delta=runtime_config.delta,
                    mode="mean",
                )
        for x, y in zip(
            feature_group_attribution_scores_batch_list_1,
            feature_group_attribution_scores_batch_list_2,
        ):
            for xx, yy in zip(x, y):
                assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
