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
        for max_examples_per_batch in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.basic_model_assert(
                **self.basic_single_setup(),
                expected=torch.tensor([True]),
                max_examples_per_batch=max_examples_per_batch,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, [x.float() for x in fwds_per_run])

    def test_basic_batch(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_examples_per_batch in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.basic_model_assert(
                **self.basic_batch_setup(),
                expected=torch.tensor([True] * 3),
                max_examples_per_batch=max_examples_per_batch,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, [x.float() for x in fwds_per_run])

    def test_basic_additional_forward_args1(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_examples_per_batch in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.basic_model_assert(
                **self.basic_additional_forward_args_setup(),
                expected=torch.tensor([False]),
                max_examples_per_batch=max_examples_per_batch,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, [x.float() for x in fwds_per_run])

    def test_classification_convnet_multi_targets(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_examples_per_batch in [
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
                max_examples_per_batch=max_examples_per_batch,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, [x.float() for x in fwds_per_run])

    def test_classification_tpl_target(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_examples_per_batch in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.basic_model_assert(
                **self.classification_tpl_target_setup(),
                expected=torch.tensor([True] * 4),
                max_examples_per_batch=max_examples_per_batch,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, [x.float() for x in fwds_per_run])

    def test_classification_tpl_target_w_baseline_perturb(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_examples_per_batch in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.basic_model_assert(
                **self.classification_tpl_target_w_baseline_setup(),
                expected=torch.tensor([True] * 4),
                max_examples_per_batch=max_examples_per_batch,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, [x.float() for x in fwds_per_run])

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
        max_examples_per_batch: int = None,
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
            feature_masks=feature_masks,
            additional_forward_args=additional_forward_args,
            target=target,
            max_examples_per_batch=max_examples_per_batch,
        )
        return score

    def monotonicity_corr_assert(
        self,
        expected: Tensor,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        attributions: TensorOrTupleOfTensorsGeneric,
        baselines: Optional[BaselineType] = None,
        feature_masks: TensorOrTupleOfTensorsGeneric = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        max_examples_per_batch: int = None,
    ) -> Tensor:
        mono, fwds = monotonicity(
            forward_func=model,
            inputs=inputs,
            attributions=attributions,
            baselines=baselines,
            feature_masks=feature_masks,
            additional_forward_args=additional_forward_args,
            target=target,
            max_examples_per_batch=max_examples_per_batch,
        )
        attributions, _ = _tuple_tensors_to_tensors(attributions)
        self.assertEqual(fwds.shape[0], attributions.shape[0])  # match batch size
        self.assertEqual(
            fwds[0].numel(), attributions[0].numel()
        )  # match number of features
        assertTensorAlmostEqual(self, mono.float(), expected.float())
        return mono, fwds
