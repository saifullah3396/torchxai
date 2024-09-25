import logging
import unittest
from logging import getLogger
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from torch import Tensor
from torch.nn import Module

from tests.helpers.basic import (
    assertAllTensorsAreAlmostEqualWithNan,
    assertTensorAlmostEqualWithNan,
    set_all_random_seeds,
)
from tests.metrics.base import MetricTestsBase
from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.metrics._utils.perturbation import (
    default_random_perturb_func,
    default_zero_baseline_func,
)
from torchxai.metrics.faithfulness.faithfulness_corr import faithfulness_corr

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class Test(MetricTestsBase):
    def test_basic_single(self) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        expected_outputs = [0.9433, 0.9429, 0.9433]
        for max_examples_per_batch, expected_output in zip(
            [
                5,
                1,
                40,
            ],
            expected_outputs,
        ):
            set_all_random_seeds(1234)
            self.output_assert(
                **self.basic_single_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([expected_output]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
            )

    def test_basic_single_zero_baseline(self) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )

        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:

            set_all_random_seeds(1234)
            faithfulness, attribution_sums, perturbation_fwds = self.output_assert(
                **self.basic_single_setup(),
                # with zero baseline all runs with different max_examples_per_batch should result in the the same output
                perturb_func=default_zero_baseline_func(),
                expected=torch.tensor([0.9429]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_basic_single_fixed_baseline(self) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )

        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:

            set_all_random_seeds(1234)
            faithfulness, attribution_sums, perturbation_fwds = self.output_assert(
                **self.basic_single_setup(),
                # with zero baseline all runs with different max_examples_per_batch should result in the the same output
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([0.9429]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                set_zero_baseline=True,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_basic_batch(self) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        expected_outputs = [
            [0.9129, 0.9520, 0.9498],
            [0.9129, 0.9520, 0.9498],
            [0.9129, 0.9518, 0.9506],
        ]
        for max_examples_per_batch, expected_output in zip(
            [
                5,
                1,
                40,
            ],
            expected_outputs,
        ):

            set_all_random_seeds(1234)
            self.output_assert(
                **self.basic_batch_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor(expected_output),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
            )

    def test_basic_batch_with_same_perturbation_mask_for_batch(self) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        expected_outputs = [
            [0.9129, 0.9129, 0.9129],
            [0.9129, 0.9129, 0.9129],
            [0.9129, 0.9129, 0.9129],
        ]
        for max_examples_per_batch, expected_output in zip(
            [
                5,
                1,
                40,
            ],
            expected_outputs,
        ):
            set_all_random_seeds(1234)
            self.output_assert(
                **self.basic_batch_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor(expected_output),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                set_same_perturbation_mask_for_batch=True,
            )

    def test_basic_batch_zero_baseline(self) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )

        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:

            set_all_random_seeds(1234)
            faithfulness, attribution_sums, perturbation_fwds = self.output_assert(
                **self.basic_batch_setup(),
                # with zero baseline all runs with different max_examples_per_batch should result in the the same output
                perturb_func=default_zero_baseline_func(),
                expected=torch.tensor([0.9129, 0.9517, 0.9495]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                delta=1e-3,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_basic_batch_fixed_zero_baseline(self) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )

        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:

            set_all_random_seeds(1234)
            faithfulness, attribution_sums, perturbation_fwds = self.output_assert(
                **self.basic_batch_setup(),
                # with zero baseline all runs with different max_examples_per_batch should result in the the same output
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([0.9129, 0.9517, 0.9495]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                set_zero_baseline=True,
                delta=1e-3,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_basic_batch_fixed_random_baseline(self) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )

        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:

            set_all_random_seeds(1234)
            setup_kwargs = self.basic_batch_setup()
            baselines = tuple(
                torch.tensor(
                    np.random.uniform(low=-0.02, high=0.02, size=x.shape),
                    device=x.device,
                ).float()
                for x in setup_kwargs["inputs"]
            )

            faithfulness, attribution_sums, perturbation_fwds = self.output_assert(
                **setup_kwargs,
                # with zero baseline all runs with different max_examples_per_batch should result in the the same output
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([0.9157, 0.9522, 0.9495]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                baselines=baselines,
                delta=1e-3,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_basic_additional_forward_args1(self) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            self.output_assert(
                **self.basic_additional_forward_args_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([torch.nan]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
            )

    def test_classification_convnet_multi_targets_zero_baseline_same_perturb(
        self,
    ) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            faithfulness, attribution_sums, perturbation_fwds = self.output_assert(
                **self.classification_convnet_multi_targets_setup(),
                perturb_func=default_zero_baseline_func(),
                expected=torch.tensor([0.7279] * 20),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                set_same_perturbation_mask_for_batch=True,
                delta=1e-3,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_classification_convnet_multi_targets_zero_baseline_diff_perturb(
        self,
    ) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            faithfulness, attribution_sums, perturbation_fwds = self.output_assert(
                **self.classification_convnet_multi_targets_setup(),
                perturb_func=default_zero_baseline_func(),
                expected=torch.tensor(
                    [
                        0.7279,
                        0.8380,
                        0.9166,
                        0.6253,
                        0.8569,
                        0.8390,
                        0.8085,
                        0.9469,
                        0.9338,
                        0.9071,
                        0.8843,
                        0.7677,
                        0.9719,
                        0.8147,
                        0.9169,
                        0.8926,
                        0.9064,
                        0.7807,
                        0.9118,
                        0.9217,
                    ],
                ),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                set_same_perturbation_mask_for_batch=False,
                delta=1e-3,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_classification_convnet_multi_targets_rand_baseline_same_perturb(
        self,
    ) -> None:
        expected_outputs = [
            [
                0.7279,
                0.7286,
                0.7283,
                0.7278,
                0.7287,
                0.7284,
                0.7285,
                0.7278,
                0.7285,
                0.7283,
                0.7272,
                0.7277,
                0.7275,
                0.7276,
                0.7292,
                0.7279,
                0.7279,
                0.7281,
                0.7274,
                0.7278,
            ],
            [
                0.7279,
                0.7286,
                0.7283,
                0.7278,
                0.7287,
                0.7284,
                0.7285,
                0.7278,
                0.7285,
                0.7283,
                0.7272,
                0.7277,
                0.7275,
                0.7276,
                0.7292,
                0.7279,
                0.7279,
                0.7281,
                0.7274,
                0.7278,
            ],
            [
                0.7281,
                0.7283,
                0.7283,
                0.7286,
                0.7286,
                0.7272,
                0.7281,
                0.7287,
                0.7282,
                0.7273,
                0.7281,
                0.7280,
                0.7272,
                0.7270,
                0.7282,
                0.7269,
                0.7281,
                0.7281,
                0.7278,
                0.7280,
            ],
        ]
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch, expected_output in zip(
            [
                5,
                1,
                40,
            ],
            expected_outputs,
        ):
            set_all_random_seeds(1234)
            self.output_assert(
                **self.classification_convnet_multi_targets_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor(expected_output),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                set_same_perturbation_mask_for_batch=True,
                delta=1e-3,
            )

    def test_classification_convnet_multi_targets_rand_baseline_diff_perturb(
        self,
    ) -> None:
        expected_outputs = [
            # faithfulness is very sensitive to perturbations, even for same inputs the faithfulness can vary a lot
            [
                0.7279,
                0.8383,
                0.9167,
                0.6278,
                0.8565,
                0.8392,
                0.8088,
                0.9471,
                0.9336,
                0.9071,
                0.8843,
                0.7678,
                0.9719,
                0.8148,
                0.9168,
                0.8926,
                0.9063,
                0.7806,
                0.9117,
                0.9214,
            ],
            [
                0.7279,
                0.8383,
                0.9167,
                0.6278,
                0.8565,
                0.8392,
                0.8088,
                0.9471,
                0.9336,
                0.9071,
                0.8843,
                0.7678,
                0.9719,
                0.8148,
                0.9168,
                0.8926,
                0.9063,
                0.7806,
                0.9117,
                0.9214,
            ],
            [
                0.7281,
                0.8381,
                0.9167,
                0.6260,
                0.8568,
                0.8391,
                0.8084,
                0.9471,
                0.9341,
                0.9072,
                0.8844,
                0.7677,
                0.9719,
                0.8151,
                0.9172,
                0.8927,
                0.9064,
                0.7811,
                0.9116,
                0.9217,
            ],
        ]
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch, expected_output in zip(
            [
                5,
                1,
                40,
            ],
            expected_outputs,
        ):
            set_all_random_seeds(1234)
            self.output_assert(
                **self.classification_convnet_multi_targets_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor(expected_output),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                set_same_perturbation_mask_for_batch=False,
                delta=1e-3,
            )

    def test_classification_tpl_target_zero_baseline_same_perturb(self) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            faithfulness, attribution_sums, perturbation_fwds = self.output_assert(
                **self.classification_tpl_target_setup(),
                perturb_func=default_zero_baseline_func(),
                expected=torch.tensor([1.000, 1.0000, 1.0000, 1.0000]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                delta=1e-3,
                set_same_perturbation_mask_for_batch=True,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_classification_tpl_target_zero_baseline_diff_perturb(self) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            faithfulness, attribution_sums, perturbation_fwds = self.output_assert(
                **self.classification_tpl_target_setup(),
                perturb_func=default_zero_baseline_func(),
                expected=torch.tensor([1.000, 1.0000, 1.0000, 1.0000]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                delta=1e-3,
                set_same_perturbation_mask_for_batch=False,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_classification_tpl_target_rand_baseline_same_perturb(self) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            self.output_assert(
                **self.classification_tpl_target_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([1.000, 1.0000, 1.0000, 1.0000]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                delta=1e-3,
                set_same_perturbation_mask_for_batch=True,
            )

    def test_classification_tpl_target_rand_baseline_diff_perturb(self) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            self.output_assert(
                **self.classification_tpl_target_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([1.000, 1.0000, 1.0000, 1.0000]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                delta=1e-3,
                set_same_perturbation_mask_for_batch=False,
            )

    def test_classification_tpl_target_w_baseline_zero_baseline_same_perturb(
        self,
    ) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            faithfulness, attribution_sums, perturbation_fwds = self.output_assert(
                **self.classification_tpl_target_w_baseline_setup(),
                perturb_func=default_zero_baseline_func(),
                expected=torch.tensor([0.9716, 0.9988, 0.9995, 1.0000]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                delta=1e-3,
                set_same_perturbation_mask_for_batch=True,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_classification_tpl_target_w_baseline_zero_baseline_diff_perturb(
        self,
    ) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            faithfulness, attribution_sums, perturbation_fwds = self.output_assert(
                **self.classification_tpl_target_w_baseline_setup(),
                perturb_func=default_zero_baseline_func(),
                expected=torch.tensor([0.9716, 0.9988, 0.9995, 1.0000]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                delta=1e-3,
                set_same_perturbation_mask_for_batch=False,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_classification_tpl_target_w_baseline_rand_baseline_same_perturb(
        self,
    ) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            self.output_assert(
                **self.classification_tpl_target_w_baseline_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([0.9998, 1.0000, 1.0000, 1.0000]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                delta=1e-3,
                set_same_perturbation_mask_for_batch=True,
            )

    def test_classification_tpl_target_w_baseline_rand_baseline_diff_perturb(
        self,
    ) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_examples_per_batch in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            self.output_assert(
                **self.classification_tpl_target_w_baseline_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([0.9998, 1.0000, 1.0000, 1.0000]),
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=0.5,
                n_perturb_samples=10,
                delta=1e-3,
                set_same_perturbation_mask_for_batch=False,
            )

    def output_assert(
        self,
        expected: Tensor,
        explainer: Union[Attribution, FusionExplainer],
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        feature_mask: TensorOrTupleOfTensorsGeneric = None,
        baselines: BaselineType = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        perturb_func: Callable = default_random_perturb_func(),
        n_perturb_samples: int = 10,
        max_examples_per_batch: int = None,
        perturbation_probability: float = 0.1,
        multiply_by_inputs: bool = False,
        delta: float = 1.0e-4,
        set_zero_baseline: bool = False,
        set_same_perturbation_mask_for_batch: bool = False,
    ) -> Tensor:
        if baselines is None and set_zero_baseline:
            if isinstance(inputs, tuple):
                baselines = tuple(torch.zeros_like(x).float() for x in inputs)
            else:
                baselines = torch.zeros_like(inputs).float()

        explanations = self.compute_explanations(
            explainer,
            inputs,
            additional_forward_args,
            baselines,
            target,
            multiply_by_inputs,
        )
        if isinstance(inputs, tuple):
            bsz = inputs[0].shape[0]
        else:
            bsz = inputs.shape[0]
        faithfulness_corr_score, attribution_sums, perturbation_fwds = (
            faithfulness_corr(
                forward_func=model,
                inputs=inputs,
                attributions=explanations,
                baselines=baselines,
                feature_mask=feature_mask,
                additional_forward_args=additional_forward_args,
                target=target,
                perturb_func=perturb_func,
                n_perturb_samples=n_perturb_samples,
                max_examples_per_batch=max_examples_per_batch,
                perturbation_probability=perturbation_probability,
                set_same_perturbation_mask_for_batch=set_same_perturbation_mask_for_batch,
            )
        )
        self.assertListEqual(
            list(attribution_sums.shape), list(perturbation_fwds.shape)
        )  # match batch size
        self.assertListEqual(
            list(attribution_sums.shape), [bsz, n_perturb_samples]
        )  # match batch size
        assertTensorAlmostEqualWithNan(
            self, faithfulness_corr_score.float(), expected.float(), delta=delta
        )
        return faithfulness_corr_score, attribution_sums, perturbation_fwds


if __name__ == "__main__":
    unittest.main()
