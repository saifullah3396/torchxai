import logging
from logging import getLogger
from typing import Any, Optional, cast

import torch
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from torch import Tensor
from torch.nn import Module

from tests.helpers.basic import assertTensorAlmostEqual
from tests.metrics.base import MetricTestsBase
from torchxai.metrics.axiomatic.completeness import completeness

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class Test(MetricTestsBase):
    def test_basic_single(self) -> None:
        self.basic_model_assert(**self.basic_single_setup(), expected=torch.zeros(1))

    def test_basic_batch(self) -> None:
        self.basic_model_assert(**self.basic_batch_setup(), expected=torch.zeros(3))

    def test_basic_additional_forward_args1(self) -> None:
        self.basic_model_assert(
            **self.basic_additional_forward_args_setup(),
            expected=torch.zeros(1),
        )

    def test_classification_convnet_multi_targets(self) -> None:
        self.basic_model_assert(
            **self.classification_convnet_multi_targets_setup(),
            expected=torch.zeros(20),
        )

    def test_classification_tpl_target(self) -> None:
        self.basic_model_assert(
            **self.classification_tpl_target_setup(),
            expected=torch.tensor([0.6538, 0, 0, 0]),
        )

    def test_classification_tpl_target_w_baseline(self) -> None:
        self.basic_model_assert(
            **self.classification_tpl_target_w_baseline_setup(),
            expected=torch.tensor([0.3269, 0, 0, 0]),
        )

    def basic_model_assert(
        self,
        attribution_fn: Attribution,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        expected: Tensor,
        additional_forward_args: Optional[Any] = None,
        baselines: Optional[BaselineType] = None,
        target: Optional[TargetType] = None,
        multiply_by_inputs: bool = False,
    ) -> Tensor:
        attributions = attribution_fn.attribute(
            inputs,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            target=target,
        )
        if multiply_by_inputs:
            attributions = cast(
                TensorOrTupleOfTensorsGeneric,
                tuple(attr / input for input, attr in zip(inputs, attributions)),
            )

        score = self.completeness_assert(
            expected=expected,
            model=model,
            inputs=inputs,
            attributions=attributions,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            target=target,
        )
        return score

    def completeness_assert(
        self,
        expected: Tensor,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        attributions: TensorOrTupleOfTensorsGeneric,
        baselines: Optional[BaselineType] = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
    ) -> Tensor:
        score = completeness(
            forward_func=model,
            inputs=inputs,
            attributions=attributions,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            target=target,
        )
        assertTensorAlmostEqual(self, score, expected)
        return score
