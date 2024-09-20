import logging
from logging import getLogger
from typing import Any, Callable, Optional, cast

import torch
from captum._utils.typing import (BaselineType, TargetType,
                                  TensorOrTupleOfTensorsGeneric)
from captum.attr import Attribution
from torch import Tensor
from torch.nn import Module

from tests.helpers.basic import (assertAllTensorsAreAlmostEqualWithNan,
                                 assertTensorAlmostEqual)
from tests.metrics.base import MetricTestsBase
from torchxai.metrics.axiomatic.monotonicity_corr_and_non_sens import (
    default_random_perturb_func, monotonicity_corr_and_non_sens)

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
                perturb_func=default_random_perturb_func(),
                expected=torch.zeros(1),
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
                perturb_func=default_random_perturb_func(),
                expected=torch.zeros(3),
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
                perturb_func=default_random_perturb_func(),
                expected=torch.ones(1),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_example=max_features_processed_per_example,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_classification_convnet_multi_targets(self) -> None:
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
                **self.classification_convnet_multi_targets_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([4] * 20),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_example=max_features_processed_per_example,
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
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([0, 0, 0, 0]),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_example=max_features_processed_per_example,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_classification_tpl_target_w_baseline(self) -> None:
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
                **self.classification_tpl_target_w_baseline_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([1, 0, 0, 0]),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_example=max_features_processed_per_example,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def basic_model_assert(
        self,
        expected: Tensor,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        attribution_fn: Attribution,
        feature_mask: TensorOrTupleOfTensorsGeneric = None,
        baselines: BaselineType = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        perturb_func: Callable = default_random_perturb_func(),
        n_perturbations_per_feature: int = 10,
        max_features_processed_per_example: int = None,
        multiply_by_inputs: bool = False,
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
        score = self.non_sensitivity_assert(
            expected=expected,
            model=model,
            inputs=inputs,
            attributions=attributions,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            target=target,
            perturb_func=perturb_func,
            n_perturbations_per_feature=n_perturbations_per_feature,
            max_features_processed_per_example=max_features_processed_per_example,
        )
        return score

    def non_sensitivity_assert(
        self,
        expected: Tensor,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        attributions: TensorOrTupleOfTensorsGeneric,
        feature_mask: TensorOrTupleOfTensorsGeneric = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        perturb_func: Callable = default_random_perturb_func(),
        n_perturbations_per_feature: int = 10,
        max_features_processed_per_example: int = None,
    ) -> Tensor:
        _, non_sens, _ = monotonicity_corr_and_non_sens(
            forward_func=model,
            inputs=inputs,
            attributions=attributions,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            target=target,
            perturb_func=perturb_func,
            n_perturbations_per_feature=n_perturbations_per_feature,
            max_features_processed_per_example=max_features_processed_per_example,
        )
        assertTensorAlmostEqual(self, non_sens, expected)
        return non_sens
