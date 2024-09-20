import logging
from logging import getLogger
from typing import Any, Optional, cast

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
from torchxai.metrics.faithfulness.faithfulness_estimate import faithfulness_estimate

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class Test(MetricTestsBase):
    def test_basic_single(self) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_features_processed_per_example in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            self.basic_model_assert(
                **self.basic_single_setup(),
                expected=torch.tensor([1]),
                max_features_processed_per_example=max_features_processed_per_example,
            )

    def test_basic_batch(self) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_features_processed_per_example in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            self.basic_model_assert(
                **self.basic_batch_setup(),
                expected=torch.tensor([1, 1, 1]),
                max_features_processed_per_example=max_features_processed_per_example,
            )

    def test_basic_additional_forward_args1(self) -> None:
        faithfulness_per_run, attribution_sums_per_run, perturbation_fwds_per_run = (
            [],
            [],
            [],
        )
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_features_processed_per_example in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            faithfulness, attribution_sums, perturbation_fwds = self.basic_model_assert(
                **self.basic_additional_forward_args_setup(),
                expected=torch.tensor([torch.nan]),
                max_features_processed_per_example=max_features_processed_per_example,
            )
            faithfulness_per_run.append(faithfulness)
            attribution_sums_per_run.append(attribution_sums)
            perturbation_fwds_per_run.append(perturbation_fwds)
        assertAllTensorsAreAlmostEqualWithNan(self, faithfulness_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, attribution_sums_per_run)
        assertAllTensorsAreAlmostEqualWithNan(self, perturbation_fwds_per_run)

    def test_classification_convnet_multi_targets(
        self,
    ) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_features_processed_per_example in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            self.basic_model_assert(
                **self.classification_convnet_multi_targets_setup(),
                expected=torch.tensor([0.4150] * 20),
                max_features_processed_per_example=max_features_processed_per_example,
                delta=1e-3,
            )

    def test_classification_tpl_target(self) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_features_processed_per_example in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            self.basic_model_assert(
                **self.classification_tpl_target_setup(),
                expected=torch.tensor([0.9966, 1.0000, 1.0000, 1.0000]),
                max_features_processed_per_example=max_features_processed_per_example,
                delta=1e-3,
            )

    def test_classification_tpl_target_w_baseline(
        self,
    ) -> None:
        # slight difference in expected output due to randomness in the default_perturb_func
        for max_features_processed_per_example in [
            5,
            1,
            40,
        ]:
            set_all_random_seeds(1234)
            self.basic_model_assert(
                **self.classification_tpl_target_w_baseline_setup(),
                expected=torch.tensor([1.0000, 1.0000, 1.0000, 1.0000]),
                max_features_processed_per_example=max_features_processed_per_example,
                delta=1e-3,
            )

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
        max_features_processed_per_example: int = None,
        multiply_by_inputs: bool = False,
        delta: float = 1.0e-4,
        set_zero_baseline: bool = False,
    ) -> Tensor:
        if baselines is None and set_zero_baseline:
            if isinstance(inputs, tuple):
                baselines = tuple(torch.zeros_like(x).float() for x in inputs)
            else:
                baselines = torch.zeros_like(inputs).float()

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

        score = self.faithfulness_estimate_assert(
            expected=expected,
            model=model,
            inputs=inputs,
            attributions=attributions,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            target=target,
            max_features_processed_per_example=max_features_processed_per_example,
            delta=delta,
        )
        return score

    def faithfulness_estimate_assert(
        self,
        expected: Tensor,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        attributions: TensorOrTupleOfTensorsGeneric,
        baselines: Optional[BaselineType] = None,
        feature_mask: TensorOrTupleOfTensorsGeneric = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        max_features_processed_per_example: int = None,
        delta: float = 1.0e-4,
    ) -> Tensor:
        output, attributions_sum_perturbed, inputs_perturbed_fwd_diffs = (
            faithfulness_estimate(
                forward_func=model,
                inputs=inputs,
                attributions=attributions,
                baselines=baselines,
                feature_mask=feature_mask,
                additional_forward_args=additional_forward_args,
                target=target,
                max_features_processed_per_example=max_features_processed_per_example,
            )
        )
        assertTensorAlmostEqualWithNan(
            self, output.float(), expected.float(), delta=delta
        )
        return output, attributions_sum_perturbed, inputs_perturbed_fwd_diffs
