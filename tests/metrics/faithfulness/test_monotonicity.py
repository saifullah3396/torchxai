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
    assertTensorAlmostEqual,
)
from tests.metrics.base import MetricTestsBase
from torchxai.metrics._utils.common import _tuple_tensors_to_tensors
from torchxai.metrics.faithfulness.monotonicity import monotonicity

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class Test(MetricTestsBase):
    def test_basic_single(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_features_processed_per_example in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.basic_model_assert(
                **self.basic_single_setup(),
                expected=torch.tensor([True]),
                max_features_processed_per_example=max_features_processed_per_example,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, fwds_per_run)

    def test_basic_batch(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_features_processed_per_example in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.basic_model_assert(
                **self.basic_batch_setup(),
                expected=torch.tensor([True] * 3),
                max_features_processed_per_example=max_features_processed_per_example,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, fwds_per_run)

    def test_basic_additional_forward_args1(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_features_processed_per_example in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.basic_model_assert(
                **self.basic_additional_forward_args_setup(),
                expected=torch.tensor([False]),
                max_features_processed_per_example=max_features_processed_per_example,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, fwds_per_run)

    def test_classification_convnet_multi_targets(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_features_processed_per_example in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.basic_model_assert(
                **self.classification_convnet_multi_targets_setup(),
                expected=torch.tensor(
                    [True] * 20,
                    dtype=torch.float64,
                ),
                max_features_processed_per_example=max_features_processed_per_example,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, fwds_per_run)

    def test_classification_tpl_target(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_features_processed_per_example in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.basic_model_assert(
                **self.classification_tpl_target_setup(),
                expected=torch.tensor([True] * 4),
                max_features_processed_per_example=max_features_processed_per_example,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, fwds_per_run)

    def test_classification_tpl_target_w_baseline_perturb(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_features_processed_per_example in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.basic_model_assert(
                **self.classification_tpl_target_w_baseline_setup(),
                expected=torch.tensor([True] * 4),
                max_features_processed_per_example=max_features_processed_per_example,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, fwds_per_run)

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
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            target=target,
            max_features_processed_per_example=max_features_processed_per_example,
        )
        return score

    def monotonicity_corr_assert(
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
    ) -> Tensor:
        mono, fwds = monotonicity(
            forward_func=model,
            inputs=inputs,
            attributions=attributions,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            target=target,
            max_features_processed_per_example=max_features_processed_per_example,
        )
        attributions, _ = _tuple_tensors_to_tensors(attributions)
        self.assertEqual(len(fwds), attributions.shape[0])  # match batch size
        for fwd, attribution in zip(fwds, attributions):
            self.assertEqual(
                fwd.numel(), attribution.numel()
            )  # match number of features
        assertTensorAlmostEqual(self, mono.float(), expected.float())
        return mono, fwds
