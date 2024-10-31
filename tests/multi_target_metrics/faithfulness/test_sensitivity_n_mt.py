import dataclasses

import pytest
import torch  # noqa

from tests.utils.common import (
    assert_tensor_almost_equal,
    grid_segmenter,
    set_all_random_seeds,
)
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics import sensitivity_n


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    set_image_feature_mask: bool = True
    n_perturb_samples: int = 10
    max_examples_per_batch: int = None
    normalize: bool = True
    sensitivity_n: int = 1
    baselines: int = 0


test_configurations = [
    *[
        MetricTestRuntimeConfig_(
            test_name="classification_alexnet_model",
            target_fixture="classification_alexnet_model_config",
            explainer="saliency",
            override_target=[0, 1, 2],
            explainer_kwargs={"is_multi_target": True},
            delta=1e-8,
            max_examples_per_batch=[5, 1, 40],
            sensitivity_n=n,
        )
        for n in [1, 10, 20]
    ],
    *[
        MetricTestRuntimeConfig_(
            test_name="classification_alexnet_model",
            target_fixture="classification_alexnet_model_config",
            explainer="integrated_gradients",
            override_target=[0, 1, 2],
            explainer_kwargs={"is_multi_target": True},
            delta=1e-8,
            max_examples_per_batch=[5, 1, 40],
            sensitivity_n=n,
        )
        for n in [1, 10, 20]
    ],
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

    runtime_config.max_examples_per_batch = _format_to_list(
        runtime_config.max_examples_per_batch
    )

    for max_examples_per_batch in runtime_config.max_examples_per_batch:
        set_all_random_seeds(1234)
        sensitivity_n_batch_list_1 = sensitivity_n(
            n=runtime_config.sensitivity_n,
            forward_func=base_config.model,
            inputs=base_config.inputs,
            attributions=explanations,
            baselines=runtime_config.baselines,
            feature_mask=base_config.feature_mask,
            additional_forward_args=base_config.additional_forward_args,
            target=base_config.target,
            n_perturb_samples=runtime_config.n_perturb_samples,
            max_examples_per_batch=max_examples_per_batch,
            normalize=runtime_config.normalize,
            is_multi_target=True,
        )

        sensitivity_n_batch_list_2 = []
        for explanation, target in zip(explanations, runtime_config.override_target):
            set_all_random_seeds(1234)
            sensitivity_n_batch = sensitivity_n(
                n=runtime_config.sensitivity_n,
                forward_func=base_config.model,
                inputs=base_config.inputs,
                attributions=explanation,
                baselines=runtime_config.baselines,
                feature_mask=base_config.feature_mask,
                additional_forward_args=base_config.additional_forward_args,
                target=target,
                n_perturb_samples=runtime_config.n_perturb_samples,
                max_examples_per_batch=max_examples_per_batch,
                normalize=runtime_config.normalize,
            )
            sensitivity_n_batch_list_2.append(sensitivity_n_batch)

        assert len(sensitivity_n_batch_list_1) == len(sensitivity_n_batch_list_2)
        for x, y in zip(sensitivity_n_batch_list_1, sensitivity_n_batch_list_2):
            for xx, yy in zip(x, y):
                assert_tensor_almost_equal(
                    xx.float(), yy.float(), delta=runtime_config.delta, mode="mean"
                )
