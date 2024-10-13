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
    perturbation_probability: float = 0.5
    set_same_perturbation_mask_for_batch: bool = False
    assert_across_runs: bool = True
    set_fixed_baseline_of_type: bool = False
    device: str = "cpu"


test_configurations = [
    MetricTestRuntimeConfig_(
        test_name="basic_model_single_input_config_integrated_gradients_random_baseline_fn",
        target_fixture="basic_model_single_input_config",
        expected=[
            # slight differences across runs expected due to difference in per-batch randomness
            torch.tensor([x])
            for x in [0.9433, 0.9429, 0.9433, 0.9537, 0.9538, 0.9537]
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
        test_name="basic_model_single_input_config_integrated_gradients_zero_baseline_fn",
        target_fixture="basic_model_single_input_config",
        expected=torch.tensor(
            [0.9429]
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
        test_name="basic_model_single_input_config_integrated_gradients_random_baseline_fn_with_fixed_zero_baseline",
        target_fixture="basic_model_single_input_config",
        expected=torch.tensor(
            [0.9429]
        ),  # slight differences across runs not expected as zero baseline is used
        perturb_func=default_random_perturb_func(),  # here we use random function but use the underlying fixed baseline
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
        test_name="basic_model_single_input_config_integrated_gradients_random_baseline_fn_with_fixed_random_baseline",
        target_fixture="basic_model_single_input_config",
        expected=torch.tensor(
            [0.9563]
        ),  # slight differences across runs not expected as zero baseline is used
        perturb_func=default_random_perturb_func(),  # here we use random function but use the underlying fixed baseline
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
        test_name="basic_model_batch_input_config_integrated_gradients_random_baseline_fn",
        target_fixture="basic_model_batch_input_config",
        expected=[
            torch.tensor(
                [0.9129, 0.9520, 0.9498]
            ),  # 10 perturbations per example, run 1
            torch.tensor(
                [0.9129, 0.9520, 0.9498]
            ),  # 10 perturbations per example, run 2
            torch.tensor(
                [0.9129, 0.9518, 0.9506]
            ),  # 10 perturbations per example, run 3
            torch.tensor(
                [0.9551, 0.9517, 0.9485]
            ),  # 100 perturbations per example, run 4
            torch.tensor(
                [0.9551, 0.9517, 0.9485]
            ),  # 100 perturbations per example, run 5
            torch.tensor(
                [0.9551, 0.9517, 0.9483]
            ),  # 100 perturbations per example, run 6
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
        test_name="basic_model_batch_input_config_integrated_gradients_random_baseline_fn_with_same_perturbation_mask_across_batch",
        target_fixture="basic_model_batch_input_config",
        expected=[
            # since set_same_perturbation_mask_for_batch, the perturbation mask will be the same across batch and
            # so all inputs which have same values should return the same outputs
            torch.tensor(
                [0.9129, 0.9129, 0.9129]
            ),  # 10 perturbations per example, run 1
            torch.tensor(
                [0.9129, 0.9129, 0.9129]
            ),  # 10 perturbations per example, run 2
            torch.tensor(
                [0.9129, 0.9129, 0.9129]
            ),  # 10 perturbations per example, run 3
        ],
        n_perturb_samples=[
            10,
            10,
            10,
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=True,
        set_same_perturbation_mask_for_batch=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_config_integrated_gradients_zero_baseline_fn",
        target_fixture="basic_model_batch_input_config",
        expected=[
            # since zero_baseline is set it will override random baseline in default_random_perturb_func,
            # so the outputs should be same across runs
            torch.tensor(
                [0.9129, 0.9517, 0.9495]
            ),  # 10 perturbations per example, run 1
            torch.tensor(
                [0.9129, 0.9517, 0.9495]
            ),  # 10 perturbations per example, run 2
            torch.tensor(
                [0.9129, 0.9517, 0.9495]
            ),  # 10 perturbations per example, run 3
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
        test_name="basic_model_batch_input_config_integrated_gradients_random_baseline_fn_with_fixed_zero_baseline",
        target_fixture="basic_model_batch_input_config",
        expected=[
            # since zero_baseline is set it will override random baseline in default_random_perturb_func,
            # so the outputs should be same across runs
            torch.tensor(
                [0.9129, 0.9517, 0.9495]
            ),  # 10 perturbations per example, run 1
            torch.tensor(
                [0.9129, 0.9517, 0.9495]
            ),  # 10 perturbations per example, run 2
            torch.tensor(
                [0.9129, 0.9517, 0.9495]
            ),  # 10 perturbations per example, run 3
        ],
        n_perturb_samples=[
            10,
            10,
            10,
        ],
        max_examples_per_batch=[5, 1, 40],
        perturb_func=default_random_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="zero",
        assert_across_runs=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_config_integrated_gradients_random_baseline_fn_with_fixed_random_baseline",
        target_fixture="basic_model_batch_input_config",
        expected=[
            # since zero_baseline is set it will override random baseline in default_random_perturb_func,
            # so the outputs should be same across runs
            torch.tensor(
                [0.9129, 0.9561, 0.9706]
            ),  # 10 perturbations per example, run 1
            torch.tensor(
                [0.9129, 0.9561, 0.9706]
            ),  # 10 perturbations per example, run 2
            torch.tensor(
                [0.9129, 0.9561, 0.9706]
            ),  # 10 perturbations per example, run 3
        ],
        n_perturb_samples=[
            10,
            10,
            10,
        ],
        max_examples_per_batch=[5, 1, 40],
        perturb_func=default_random_perturb_func(),  # here we use random function but use the underlying fixed baseline
        set_fixed_baseline_of_type="random",
        assert_across_runs=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="basic_model_batch_input_with_additional_forward_args_config_integrated_gradients",
        target_fixture="basic_model_batch_input_with_additional_forward_args_config",
        expected=torch.tensor([torch.nan]),
        n_perturb_samples=[10, 10, 10],
        max_examples_per_batch=[5, 1, 40],
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients_random_baseline_fn",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # slight differences across runs expected due to difference in per-batch randomness
            torch.tensor(x)
            for x in [
                # faithfulness is very sensitive to perturbations, even for same inputs the faithfulness can vary a lot
                [
                    0.8897,
                    0.8151,
                    0.8823,
                    0.8943,
                    0.8021,
                    0.8323,
                    0.9283,
                    0.9047,
                    0.9337,
                    0.9381,
                    0.8780,
                    0.7656,
                    0.9322,
                    0.8673,
                    0.9053,
                    0.9184,
                    0.9401,
                    0.8492,
                    0.7850,
                    0.9526,
                ],
                [
                    0.8897,
                    0.8151,
                    0.8823,
                    0.8943,
                    0.8021,
                    0.8323,
                    0.9283,
                    0.9047,
                    0.9337,
                    0.9381,
                    0.8780,
                    0.7656,
                    0.9322,
                    0.8673,
                    0.9053,
                    0.9184,
                    0.9401,
                    0.8492,
                    0.7850,
                    0.9526,
                ],
                [
                    0.8897,
                    0.8150,
                    0.8824,
                    0.8943,
                    0.8019,
                    0.8326,
                    0.9282,
                    0.9047,
                    0.9341,
                    0.9380,
                    0.8780,
                    0.7646,
                    0.9324,
                    0.8668,
                    0.9051,
                    0.9185,
                    0.9403,
                    0.8491,
                    0.7854,
                    0.9526,
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
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients_random_baseline_fn_with_same_perturbation_mask_across_batch",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # slight differences across runs expected due to difference in per-batch randomness
            torch.tensor(x)
            for x in [
                [
                    0.8897,
                    0.8900,
                    0.8897,
                    0.8900,
                    0.8898,
                    0.8902,
                    0.8898,
                    0.8900,
                    0.8894,
                    0.8900,
                    0.8902,
                    0.8904,
                    0.8902,
                    0.8902,
                    0.8900,
                    0.8899,
                    0.8904,
                    0.8901,
                    0.8900,
                    0.8899,
                ],
                [
                    0.8897,
                    0.8900,
                    0.8897,
                    0.8900,
                    0.8898,
                    0.8902,
                    0.8898,
                    0.8900,
                    0.8894,
                    0.8900,
                    0.8902,
                    0.8904,
                    0.8902,
                    0.8902,
                    0.8900,
                    0.8899,
                    0.8904,
                    0.8901,
                    0.8900,
                    0.8899,
                ],
                [
                    0.8897,
                    0.8897,
                    0.8897,
                    0.8900,
                    0.8893,
                    0.8902,
                    0.8900,
                    0.8905,
                    0.8900,
                    0.8898,
                    0.8899,
                    0.8899,
                    0.8902,
                    0.8900,
                    0.8902,
                    0.8903,
                    0.8900,
                    0.8901,
                    0.8900,
                    0.8899,
                ],
            ]
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        set_same_perturbation_mask_for_batch=True,
        assert_across_runs=False,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients_zero_baseline_fn",
        # when mask is false the outputs should be different across batch but same across runs
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # slight differences across runs expected due to difference in per-batch randomness
            torch.tensor(x)
            for x in [
                *(
                    [
                        [
                            0.8899,
                            0.8150,
                            0.8825,
                            0.8942,
                            0.8024,
                            0.8322,
                            0.9283,
                            0.9047,
                            0.9342,
                            0.9381,
                            0.8780,
                            0.7649,
                            0.9321,
                            0.8665,
                            0.9050,
                            0.9185,
                            0.9402,
                            0.8490,
                            0.7852,
                            0.9527,
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
        set_same_perturbation_mask_for_batch=False,
        assert_across_runs=False,
        perturb_func=default_zero_baseline_func(),
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients_zero_baseline_fn_with_same_perturbation_mask_across_batch",
        # when mask is false the outputs should be same across batch and same across runs
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # slight differences across runs expected due to difference in per-batch randomness
            torch.tensor(x)
            for x in [
                [0.8899] * 20,
                [0.8899] * 20,
                [0.8899] * 20,
            ]
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        set_same_perturbation_mask_for_batch=True,
        assert_across_runs=False,
        perturb_func=default_zero_baseline_func(),
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_zero_baseline",
        # when mask is false the outputs should be different across batch but same across runs
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # slight differences across runs expected due to difference in per-batch randomness
            torch.tensor(x)
            for x in [
                *(
                    [
                        [
                            0.8899,
                            0.8150,
                            0.8825,
                            0.8942,
                            0.8024,
                            0.8322,
                            0.9283,
                            0.9047,
                            0.9342,
                            0.9381,
                            0.8780,
                            0.7649,
                            0.9321,
                            0.8665,
                            0.9050,
                            0.9185,
                            0.9402,
                            0.8490,
                            0.7852,
                            0.9527,
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
        set_same_perturbation_mask_for_batch=False,
        assert_across_runs=False,
        set_fixed_baseline_of_type="zero",
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_zero_baseline_with_same_perturbation_mask_across_batch",
        # when mask is false the outputs should be same across batch and same across runs
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # slight differences across runs expected due to difference in per-batch randomness
            torch.tensor(x)
            for x in [
                [0.8899] * 20,
                [0.8899] * 20,
                [0.8899] * 20,
            ]
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        set_same_perturbation_mask_for_batch=True,
        assert_across_runs=False,
        set_fixed_baseline_of_type="zero",
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_random_baseline",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # with fixed random baseline the perturbation masks are still different across batch, so the outputs
            # should be different across batch but same across runs
            torch.tensor(x)
            for x in [
                *(
                    [
                        [
                            0.8906,
                            0.8303,
                            0.8923,
                            0.9056,
                            0.8458,
                            0.8717,
                            0.9202,
                            0.9080,
                            0.9587,
                            0.9404,
                            0.8912,
                            0.7828,
                            0.9460,
                            0.8864,
                            0.9263,
                            0.9083,
                            0.9452,
                            0.8788,
                            0.8225,
                            0.9570,
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
        set_same_perturbation_mask_for_batch=False,
        assert_across_runs=False,
        set_fixed_baseline_of_type="random",
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_convnet_model_with_multiple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_random_baseline_with_same_perturbation_mask_across_batch",
        target_fixture="classification_convnet_model_with_multiple_targets_config",
        expected=[
            # with fixed random baseline and same perturbation masks, the outputs are still different across batch since
            # random baeline is different for each sample, so the outputs should be different across batch
            # but same across runs
            torch.tensor(x)
            for x in [
                *(
                    [
                        [
                            0.8906,
                            0.8973,
                            0.9005,
                            0.8963,
                            0.9010,
                            0.9021,
                            0.8912,
                            0.9083,
                            0.9077,
                            0.9029,
                            0.8913,
                            0.8976,
                            0.9066,
                            0.9086,
                            0.9061,
                            0.9061,
                            0.9033,
                            0.9148,
                            0.9098,
                            0.8972,
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
        set_same_perturbation_mask_for_batch=True,
        assert_across_runs=False,
        set_fixed_baseline_of_type="random",
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients_random_baseline_fn",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=[
            torch.tensor(x)
            for x in [
                [0.9980, 1.0000, 1.0000, 1.0000],
                [0.9980, 1.0000, 1.0000, 1.0000],
                [0.9980, 1.0000, 1.0000, 1.0000],
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
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients_random_baseline_fn_with_same_perturbation_mask_across_batch",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=[
            torch.tensor(x)
            for x in [
                [0.9980, 1.0000, 1.0000, 1.0000],
                [0.9980, 1.0000, 1.0000, 1.0000],
                [0.9980, 1.0000, 1.0000, 1.0000],
            ]
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_same_perturbation_mask_for_batch=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients_zero_baseline_fn",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.9980, 1.0000, 1.0000, 1.0000]),
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
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients_zero_baseline_fn_with_same_perturbation_mask_across_batch",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.9978, 1.0000, 1.0000, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_same_perturbation_mask_for_batch=True,
        perturb_func=default_zero_baseline_func(),
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_zero_baseline",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.9980, 1.0000, 1.0000, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_fixed_baseline_of_type="zero",
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_zero_baseline_with_same_perturbation_mask_across_batch",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.9978, 1.0000, 1.0000, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_same_perturbation_mask_for_batch=True,
        set_fixed_baseline_of_type="zero",
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_random_baseline",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.9299, 1.0000, 1.0000, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_fixed_baseline_of_type="random",
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_tuple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_random_baseline_with_same_perturbation_mask_across_batch",
        target_fixture="classification_multilayer_model_with_tuple_targets_config",
        expected=torch.tensor([0.9299, 1.0000, 1.0000, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_same_perturbation_mask_for_batch=True,
        set_fixed_baseline_of_type="random",
    ),
    # results can be different across runs due to randomness in perturbation
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients_random_baseline_fn",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=[
            torch.tensor(x)
            for x in [
                [0.9219, 0.9996, 0.9999, 1.0000],
                [0.9219, 0.9996, 0.9999, 1.0000],
                [0.9205, 0.9996, 0.9999, 1.0000],
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
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients_random_baseline_fn_with_same_perturbation_mask_across_batch",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=[
            torch.tensor(x)
            for x in [
                [0.9219, 0.9983, 0.9997, 0.9999],
                [0.9219, 0.9983, 0.9997, 0.9999],
                [0.9205, 0.9981, 0.9998, 0.9999],
            ]
        ],
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_same_perturbation_mask_for_batch=True,
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients_zero_baseline_fn",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([0.9196, 0.9996, 0.9999, 1.0000]),
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
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients_zero_baseline_fn_with_same_perturbation_mask_across_batch",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([0.9196, 0.9982, 1.0000, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_same_perturbation_mask_for_batch=True,
        perturb_func=default_zero_baseline_func(),
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_zero_baseline",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([0.9196, 0.9996, 0.9999, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_fixed_baseline_of_type="zero",
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_zero_baseline_with_same_perturbation_mask_across_batch",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([0.9196, 0.9982, 1.0000, 1.0000]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_same_perturbation_mask_for_batch=True,
        set_fixed_baseline_of_type="zero",
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_random_baseline",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([0.9927, 0.9995, 0.9998, 0.9999]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_fixed_baseline_of_type="random",
    ),
    MetricTestRuntimeConfig_(
        test_name="classification_multilayer_model_with_baseline_and_tuple_targets_config_integrated_gradients_random_baseline_fn_with_fixed_random_baseline_with_same_perturbation_mask_across_batch",
        target_fixture="classification_multilayer_model_with_baseline_and_tuple_targets_config",
        expected=torch.tensor([0.9927, 0.9980, 0.9997, 0.9998]),
        n_perturb_samples=[
            10,  # 10 perturbations per example, run 1
            10,  # 10 perturbations per example, run 2
            10,  # 10 perturbations per example, run 3
        ],
        max_examples_per_batch=[5, 1, 40],
        assert_across_runs=False,
        set_same_perturbation_mask_for_batch=True,
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
                perturbation_probability=runtime_config.perturbation_probability,
                set_same_perturbation_mask_for_batch=runtime_config.set_same_perturbation_mask_for_batch,
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
