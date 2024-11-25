import dataclasses
import itertools
from typing import Callable

import pytest  # noqa
import torch

from tests.utils.common import (
    assert_all_tensors_almost_equal,
    assert_tensor_almost_equal,
    set_all_random_seeds,
)
from tests.utils.containers import TestRuntimeConfig
from torchxai.metrics import faithfulness_corr
from torchxai.metrics._utils.perturbation import (
    default_fixed_baseline_perturb_func,
    default_random_perturb_func,
    default_zero_baseline_func,
)


def _format_to_list(value):
    if not isinstance(value, list):
        return [value]
    return value


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    perturb_func: Callable = default_random_perturb_func()
    n_perturb_samples: int = 10
    max_examples_per_batch: int = None
    percent_features_perturbed: float = 0.5
    assert_across_runs: bool = True
    set_fixed_baseline_of_type: bool = False
    device: str = "cpu"


test_configurations = [
    MetricTestRuntimeConfig_(
        test_name="random_baseline_fn",
        target_fixture="basic_model_single_input_config",
        expected=[
            # slight differences across runs expected due to difference in per-batch randomness
            torch.tensor([x])
            for x in [1] * 6
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
            100,  # 100 perturbations per example, run 4
            100,  # 100 perturbations per example, run 5
            100,  # 100 perturbations per example, run 6
        ],
        max_examples_per_batch=[5, 1, 40, None, 10, 40],
        assert_across_runs=False,
    ),
    MetricTestRuntimeConfig_(
        test_name="zero_baseline_fn",
        target_fixture="basic_model_single_input_config",
        expected=torch.tensor(
            [1]
        ),  # slight differences across runs not expected as zero baseline is used
        perturb_func=default_zero_baseline_func(),
        n_perturb_samples=[
            10,
            10,
            10,
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=True,  # no randomness in perturbation so no differences expected
    ),
    MetricTestRuntimeConfig_(
        test_name="fixed_zero_baseline",
        target_fixture="basic_model_single_input_config",
        expected=torch.tensor(
            [1]
        ),  # slight differences across runs not expected as zero baseline is used
        perturb_func=default_fixed_baseline_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="zero",
        n_perturb_samples=[
            10,
            10,
            10,
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=True,  # no randomness in perturbation so no differences expected
    ),
    MetricTestRuntimeConfig_(
        test_name="fixed_random_baseline",
        target_fixture="basic_model_single_input_config",
        expected=torch.tensor(
            [1]
        ),  # slight differences across runs not expected as zero baseline is used
        perturb_func=default_fixed_baseline_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="random",
        n_perturb_samples=[
            10,
            10,
            10,
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=True,  # no randomness in perturbation so no differences expected
    ),
    MetricTestRuntimeConfig_(
        test_name="random_baseline_fn",
        target_fixture="basic_model_batch_input_config",
        expected=[
            torch.tensor([1] * 3),  # 10 perturbations per example, run 1
            torch.tensor([1] * 3),  # 10 perturbations per example, run 2
            torch.tensor([1] * 3),  # 10 perturbations per example, run 3
            torch.tensor([1] * 3),  # 100 perturbations per example, run 4
            torch.tensor([1] * 3),  # 100 perturbations per example, run 5
            torch.tensor([1] * 3),  # 100 perturbations per example, run 6
        ],  # slight differences expected due to difference in per-batch randomness with different max_examples_per_batch
        n_perturb_samples=[
            10,
            10,
            10,
            100,
            100,
            100,
        ],
        max_examples_per_batch=[5, 1, 40, None, 10, 40],
        assert_across_runs=False,
    ),
    MetricTestRuntimeConfig_(
        test_name="zero_baseline_fn",
        target_fixture="basic_model_batch_input_config",
        expected=[
            # since zero_baseline is set it will override random baseline in default_random_perturb_func,
            # so the outputs should be same across runs
            torch.tensor([1] * 3),  # 10 perturbations per example, run 1
            torch.tensor([1] * 3),  # 10 perturbations per example, run 2
            torch.tensor([1] * 3),  # 10 perturbations per example, run 3
        ],
        n_perturb_samples=[
            10,
            10,
            10,
        ],
        max_examples_per_batch=[5, 1, 40],
        perturb_func=default_zero_baseline_func(),  # here we use random function but use the underlying fixed baseline
        assert_across_runs=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="fixed_zero_baseline",
        target_fixture="basic_model_batch_input_config",
        expected=[
            # since zero_baseline is set it will override random baseline in default_random_perturb_func,
            # so the outputs should be same across runs
            torch.tensor([1] * 3),  # 10 perturbations per example, run 1
            torch.tensor([1] * 3),  # 10 perturbations per example, run 2
            torch.tensor([1] * 3),  # 10 perturbations per example, run 3
        ],
        n_perturb_samples=[
            10,
            10,
            10,
        ],
        max_examples_per_batch=[5, 1, 40],
        perturb_func=default_fixed_baseline_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="zero",
        assert_across_runs=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="fixed_random_baseline",
        target_fixture="basic_model_batch_input_config",
        expected=[
            # since zero_baseline is set it will override random baseline in default_random_perturb_func,
            # so the outputs should be same across runs
            torch.tensor([1] * 3),  # 10 perturbations per example, run 1
            torch.tensor([1] * 3),  # 10 perturbations per example, run 2
            torch.tensor([1] * 3),  # 10 perturbations per example, run 3
        ],
        n_perturb_samples=[
            10,
            10,
            10,
        ],
        max_examples_per_batch=[5, 1, 40],
        perturb_func=default_fixed_baseline_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="random",
        assert_across_runs=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="random_baseline_fn",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=torch.tensor([torch.nan]),
        n_perturb_samples=[10, 10, 10],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
    ),
    MetricTestRuntimeConfig_(
        test_name="random_baseline_fn",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # slight differences across runs expected due to difference in per-batch randomness
            torch.tensor(x)
            for x in [
                # faithfulness is very sensitive to perturbations, even for same inputs the faithfulness can vary a lot
                [
                    0.7584,
                    0.8742,
                    0.9037,
                    0.6730,
                    0.8268,
                    0.8161,
                    0.9190,
                    0.7511,
                    0.7949,
                    0.5883,
                    0.9437,
                    0.8071,
                    0.7533,
                    0.8241,
                    0.8611,
                    0.5858,
                    0.7429,
                    0.7284,
                    0.8357,
                    0.8618,
                ],
                [
                    0.7584,
                    0.8742,
                    0.9037,
                    0.6730,
                    0.8268,
                    0.8161,
                    0.9190,
                    0.7511,
                    0.7949,
                    0.5883,
                    0.9437,
                    0.8071,
                    0.7533,
                    0.8241,
                    0.8611,
                    0.5858,
                    0.7429,
                    0.7284,
                    0.8357,
                    0.8618,
                ],
                [
                    0.7576,
                    0.8741,
                    0.9036,
                    0.6738,
                    0.8272,
                    0.8155,
                    0.9192,
                    0.7512,
                    0.7959,
                    0.5886,
                    0.9440,
                    0.8071,
                    0.7534,
                    0.8248,
                    0.8611,
                    0.5847,
                    0.7426,
                    0.7277,
                    0.8353,
                    0.8621,
                ],
            ]
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
    ),
    MetricTestRuntimeConfig_(
        test_name="zero_baseline_fn",
        # when mask is false the outputs should be different across batch but same across runs
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # slight differences across runs expected due to difference in per-batch randomness
            torch.tensor(x)
            for x in [
                *(
                    [
                        [
                            0.7573,
                            0.8742,
                            0.9037,
                            0.6731,
                            0.8267,
                            0.8163,
                            0.9190,
                            0.7509,
                            0.7954,
                            0.5885,
                            0.9440,
                            0.8069,
                            0.7534,
                            0.8238,
                            0.8618,
                            0.5863,
                            0.7432,
                            0.7280,
                            0.8357,
                            0.8618,
                        ]
                    ]
                    * 3
                ),
            ]
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        perturb_func=default_zero_baseline_func(),
    ),
    MetricTestRuntimeConfig_(
        test_name="fixed_zero_baseline",
        # when mask is false the outputs should be different across batch but same across runs
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # slight differences across runs expected due to difference in per-batch randomness
            torch.tensor(x)
            for x in [
                *(
                    [
                        [
                            0.7573,
                            0.8742,
                            0.9037,
                            0.6731,
                            0.8267,
                            0.8163,
                            0.9190,
                            0.7509,
                            0.7954,
                            0.5885,
                            0.9440,
                            0.8069,
                            0.7534,
                            0.8238,
                            0.8618,
                            0.5863,
                            0.7432,
                            0.7280,
                            0.8357,
                            0.8618,
                        ]
                    ]
                    * 3
                ),
            ]
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        perturb_func=default_fixed_baseline_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="zero",
    ),
    MetricTestRuntimeConfig_(
        test_name="fixed_random_baseline",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # with fixed random baseline the perturbation masks are still different across batch, so the outputs
            # should be different across batch but same across runs
            torch.tensor(x)
            for x in [
                *(
                    [
                        [
                            0.7939,
                            0.8840,
                            0.9196,
                            0.7278,
                            0.8301,
                            0.8442,
                            0.9171,
                            0.7802,
                            0.7938,
                            0.5919,
                            0.9441,
                            0.8128,
                            0.7909,
                            0.7823,
                            0.8444,
                            0.6592,
                            0.8319,
                            0.7791,
                            0.8630,
                            0.8800,
                        ]
                    ]
                    * 3
                ),
            ]
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        perturb_func=default_fixed_baseline_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="random",
    ),
    MetricTestRuntimeConfig_(
        test_name="random_baseline_fn",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=[
            torch.tensor(x)
            for x in [
                [0.9971, 1.0000, 0.9999, 1.0000],
                [0.9971, 1.0000, 0.9999, 1.0000],
                [0.9971, 1.0000, 0.9999, 1.0000],
            ]
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
    ),
    MetricTestRuntimeConfig_(
        test_name="zero_baseline_fn",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.9973, 1.0000, 1.0000, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        perturb_func=default_zero_baseline_func(),
    ),
    MetricTestRuntimeConfig_(
        test_name="fixed_zero_baseline",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.9973, 1.0000, 1.0000, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        perturb_func=default_fixed_baseline_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="zero",
    ),
    MetricTestRuntimeConfig_(
        test_name="fixed_random_baseline",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.9960, 0.9991, 0.9922, 0.9987]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        perturb_func=default_fixed_baseline_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="random",
    ),
    # results can be different across runs due to randomness in perturbation
    MetricTestRuntimeConfig_(
        test_name="random_baseline_fn",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=[
            torch.tensor(x)
            for x in [
                [0.9971, 1.0000, 0.9999, 1.0000],
                [0.9971, 1.0000, 0.9999, 1.0000],
                [0.9971, 1.0000, 0.9999, 1.0000],
            ]
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
    ),
    MetricTestRuntimeConfig_(
        test_name="zero_baseline_fn",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([0.9973, 1.0000, 1.0000, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        perturb_func=default_zero_baseline_func(),
    ),
    MetricTestRuntimeConfig_(
        test_name="fixed_zero_baseline",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([0.9973, 1.0000, 1.0000, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        perturb_func=default_fixed_baseline_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="zero",
    ),
    MetricTestRuntimeConfig_(
        test_name="fixed_random_baseline",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([0.9960, 0.9991, 0.9922, 0.9987]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        perturb_func=default_fixed_baseline_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="random",
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_faithfulness_corr(metrics_runtime_test_configuration):
    base_config, runtime_config, explanations = metrics_runtime_test_configuration
    runtime_config.n_perturb_samples = _format_to_list(runtime_config.n_perturb_samples)
    runtime_config.max_examples_per_batch = _format_to_list(
        runtime_config.max_examples_per_batch
    )
    runtime_config.expected = _format_to_list(runtime_config.expected)

    assert len(runtime_config.n_perturb_samples) == len(
        runtime_config.max_examples_per_batch
    )
    assert (
        len(runtime_config.n_perturb_samples) == len(runtime_config.expected)
        or len(runtime_config.expected) == 1
    )

    perturbation_baseline = None
    if runtime_config.set_fixed_baseline_of_type is not None:
        if runtime_config.set_fixed_baseline_of_type == "zero":
            if isinstance(base_config.inputs, tuple):
                perturbation_baseline = tuple(
                    torch.zeros_like(x) for x in base_config.inputs
                )
            else:
                perturbation_baseline = torch.zeros_like(base_config.inputs)
        elif runtime_config.set_fixed_baseline_of_type == "random":
            if isinstance(base_config.inputs, tuple):
                perturbation_baseline = tuple(
                    torch.rand_like(x) for x in base_config.inputs
                )
            else:
                perturbation_baseline = torch.rand_like(base_config.inputs)

    faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
        [],
        [],
        [],
    )
    for n_perturbs, max_examples, curr_expected in zip(
        runtime_config.n_perturb_samples,
        runtime_config.max_examples_per_batch,
        itertools.cycle(runtime_config.expected),
    ):
        set_all_random_seeds(1234)
        faithfulness_corr_score, attribution_sums, perturbation_fwds = (
            faithfulness_corr(
                forward_func=base_config.model,
                inputs=base_config.inputs,
                attributions=explanations,
                baselines=perturbation_baseline,
                feature_mask=base_config.feature_mask,
                additional_forward_args=base_config.additional_forward_args,
                target=base_config.target,
                perturb_func=runtime_config.perturb_func,
                n_perturb_samples=n_perturbs,
                max_examples_per_batch=max_examples,
                percent_features_perturbed=runtime_config.percent_features_perturbed,
                return_intermediate_results=True,
            )
        )

        if isinstance(base_config.inputs, tuple):
            bsz = base_config.inputs[0].shape[0]
        else:
            bsz = base_config.inputs.shape[0]

        assert (
            list(attribution_sums.shape)
            == list(perturbation_fwds.shape)
            == [bsz, n_perturbs]
        ), f"The size of the attribution sums and perturbation fwds must match {attribution_sums.shape} != {perturbation_fwds.shape}"

        assert_tensor_almost_equal(
            faithfulness_corr_score.float(),
            curr_expected.float(),
            delta=runtime_config.delta,
            mode="mean",
        )
        faithfulness_per_run.append(faithfulness_corr_score)
        attribution_sums_per_run.append(attribution_sums)
        perturbation_fwds_per_run.append(perturbation_fwds)
    if runtime_config.assert_across_runs:
        assert_all_tensors_almost_equal(faithfulness_per_run)
        assert_all_tensors_almost_equal(attribution_sums_per_run)
        assert_all_tensors_almost_equal(perturbation_fwds_per_run)
