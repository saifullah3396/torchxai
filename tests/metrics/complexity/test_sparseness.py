import logging
from logging import getLogger
from typing import Any, Optional, cast

import numpy as np
import torch
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from torch import Tensor
from torch.nn import Module

from tests.helpers.basic import assertTensorAlmostEqual
from tests.metrics.base import MetricTestsBase
from torchxai.metrics.complexity.sparseness import sparseness

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class Test(MetricTestsBase):
    def test_basic_single(self) -> None:
        self.basic_model_assert(
            **self.basic_single_setup(), expected=torch.tensor([0.2500]), delta=1e-3
        )

    def test_basic_batch(self) -> None:
        self.basic_model_assert(
            **self.basic_batch_setup(), expected=torch.tensor([0.2500] * 3), delta=1e-3
        )

    def test_basic_additional_forward_args1(self) -> None:
        self.basic_model_assert(
            **self.basic_additional_forward_args_setup(),
            expected=torch.tensor([0]),  # attribution here is 0 so entropy is nan
        )

    def test_classification_convnet_multi_targets(self) -> None:
        self.basic_model_assert(
            **self.classification_convnet_multi_targets_setup(),
            expected=torch.tensor([0.3840] * 20),
            delta=1.0e-3,
        )

    def test_classification_tpl_target(self) -> None:
        self.basic_model_assert(
            **self.classification_tpl_target_setup(),
            expected=torch.tensor([0.2222, 0.0889, 0.0556, 0.0404]),
            delta=1.0e-3,
        )

    def test_classification_tpl_target_w_baseline(self) -> None:
        self.basic_model_assert(
            **self.classification_tpl_target_w_baseline_setup(),
            expected=torch.tensor([0.4444, 0.1111, 0.0635, 0.0444]),
            delta=1.0e-3,
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
        delta: float = 1e-4,
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

        score = self.effective_complexity_assert(
            expected=expected,
            attributions=attributions,
            delta=delta,
        )
        return score

    def effective_complexity_assert(
        self,
        expected: Tensor,
        attributions: TensorOrTupleOfTensorsGeneric,
        delta: float = 1e-4,
    ) -> Tensor:
        output = sparseness(
            attributions=attributions,
        )
        if torch.isnan(output).all():
            assert torch.isnan(expected).all()
        else:
            assertTensorAlmostEqual(self, output, expected, delta=delta)
        return output
