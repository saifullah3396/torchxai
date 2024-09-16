import logging
from logging import getLogger
from typing import Any, Callable, Optional, cast

import numpy as np
import torch
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from cv2 import exp
from torch import Tensor
from torch.nn import Module

from tests.helpers.basic import (
    assertAllTensorsAreAlmostEqualWithNan,
    assertTensorAlmostEqual,
)
from tests.metrics.base import MetricTestsBase
from torchxai.metrics._utils.perturbation import default_perturb_func
from torchxai.metrics.axiomatic.monotonicity_corr_and_non_sens import (
    monotonicity_corr_and_non_sens,
)

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class Test(MetricTestsBase):
    def test_basic_single(self) -> None:
        outputs = []
        for n_perturbations_per_feature, max_features_processed_per_example in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            output = self.basic_model_assert(
                **self.basic_single_setup(),
                perturb_func=default_perturb_func(),
                expected=torch.ones(1),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_example=max_features_processed_per_example,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_basic_batch(self) -> None:
        outputs = []
        for n_perturbations_per_feature, max_features_processed_per_example in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            output = self.basic_model_assert(
                **self.basic_batch_setup(),
                perturb_func=default_perturb_func(),
                expected=torch.ones(3),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_example=max_features_processed_per_example,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_basic_additional_forward_args1(self) -> None:
        outputs = []
        for n_perturbations_per_feature, max_features_processed_per_example in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            output = self.basic_model_assert(
                **self.basic_additional_forward_args_setup(),
                perturb_func=default_perturb_func(),
                expected=torch.tensor([torch.nan]),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_example=max_features_processed_per_example,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_classification_convnet_multi_targets(self) -> None:
        outputs = []
        expected_outputs = [
            torch.tensor(
                [
                    0.3410,
                    0.3793,
                    0.3410,
                    0.3658,
                    0.3422,
                    0.3646,
                    0.3410,
                    0.3658,
                    0.3658,
                    0.3422,
                    0.3557,
                    0.3746,
                    0.3422,
                    0.3628,
                    0.3776,
                    0.3540,
                    0.3776,
                    0.3422,
                    0.3557,
                    0.3422,
                ],
                dtype=torch.float64,
            ),
        ]
        for (
            n_perturbations_per_feature,
            max_features_processed_per_example,
            expected_output,
        ) in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
            expected_outputs,
        ):
            output = self.basic_model_assert(
                **self.classification_convnet_multi_targets_setup(),
                perturb_func=default_perturb_func(),
                expected=expected_output,
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_example=max_features_processed_per_example,
                delta=1e-3,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_classification_tpl_target(self) -> None:
        outputs = []
        for n_perturbations_per_feature, max_features_processed_per_example in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            output = self.basic_model_assert(
                **self.classification_tpl_target_setup(),
                perturb_func=default_perturb_func(),
                expected=torch.tensor([1, 1, 1, 1]),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_example=max_features_processed_per_example,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_classification_tpl_target_w_baseline_perturb(self) -> None:
        outputs = []
        expected_outputs = [
            torch.tensor([1, 1, 1, 1]),
            torch.tensor([1, 1, 1, 1]),
            torch.tensor([1, 1, 1, 1]),
        ]
        for (
            n_perturbations_per_feature,
            max_features_processed_per_example,
            expected_output,
        ) in zip(
            [10, 1, 20],
            [
                None,
                10,
                40,
            ],
            expected_outputs,
        ):
            output = self.basic_model_assert(
                **self.classification_tpl_target_w_baseline_setup(),
                perturb_func=default_perturb_func(),
                expected=expected_output,
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_example=max_features_processed_per_example,
                delta=1e-3,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def basic_model_assert(
        self,
        expected: Tensor,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        attribution_fn: Attribution,
        feature_masks: TensorOrTupleOfTensorsGeneric = None,
        baselines: BaselineType = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        perturb_func: Callable = default_perturb_func(),
        n_perturbations_per_feature: int = 10,
        max_features_processed_per_example: int = None,
        multiply_by_inputs: bool = False,
        delta: float = 1e-4,
    ) -> Tensor:
        attributions = attribution_fn.attribute(
            inputs,
            additional_forward_args=additional_forward_args,
            target=target,
            baselines=baselines,
        )
        if multiply_by_inputs:
            attributions = cast(
                TensorOrTupleOfTensorsGeneric,
                tuple(attr / input for input, attr in zip(inputs, attributions)),
            )
        score = self.monotonicity_corr_assert(
            expected=expected,
            model=model,
            inputs=inputs,
            attributions=attributions,
            feature_masks=feature_masks,
            additional_forward_args=additional_forward_args,
            target=target,
            perturb_func=perturb_func,
            n_perturbations_per_feature=n_perturbations_per_feature,
            max_features_processed_per_example=max_features_processed_per_example,
            delta=delta,
        )
        return score

    def monotonicity_corr_assert(
        self,
        expected: Tensor,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        attributions: TensorOrTupleOfTensorsGeneric,
        feature_masks: TensorOrTupleOfTensorsGeneric = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        perturb_func: Callable = default_perturb_func(),
        n_perturbations_per_feature: int = 10,
        max_features_processed_per_example: int = None,
        delta: float = 1e-4,
    ) -> Tensor:
        output, _ = monotonicity_corr_and_non_sens(
            forward_func=model,
            inputs=inputs,
            attributions=attributions,
            feature_masks=feature_masks,
            additional_forward_args=additional_forward_args,
            target=target,
            perturb_func=perturb_func,
            n_perturbations_per_feature=n_perturbations_per_feature,
            max_features_processed_per_example=max_features_processed_per_example,
        )
        if torch.isnan(expected).all():
            assert torch.isnan(output).all()
        else:
            assertTensorAlmostEqual(self, output, expected, delta=delta)
        return output
