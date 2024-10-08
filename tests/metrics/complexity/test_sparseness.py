import logging
import unittest
from logging import getLogger
from typing import Any, Optional, Union

import torch
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from torch import Tensor
from torch.nn import Module

from tests.helpers.basic import assertTensorAlmostEqual
from tests.metrics.base import MetricTestsBase
from torchxai.explanation_framework.explainers.factory import ExplainerFactory
from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.metrics.complexity.sparseness import sparseness

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class Test(MetricTestsBase):
    def test_park_function(self) -> None:
        kwargs = self.park_function_setup()
        model = kwargs["model"]
        kwargs.pop("explainer")
        for explainer, expected in zip(
            [
                "saliency",
                "input_x_gradient",
                "integrated_gradients",
            ],
            [
                0.5501,
                0.5308,
                0.4295,
            ],  # these effective complexity results match from the paper: https://arxiv.org/pdf/2007.07584
        ):
            self.output_assert(
                **kwargs,
                explainer=ExplainerFactory.create(explainer, model),
                expected=torch.tensor([expected]),
            )

    def test_basic_single(self) -> None:
        self.output_assert(
            **self.basic_single_setup(), expected=torch.tensor([0.2500]), delta=1e-3
        )

    def test_basic_batch(self) -> None:
        self.output_assert(
            **self.basic_batch_setup(), expected=torch.tensor([0.2500] * 3), delta=1e-3
        )

    def test_basic_additional_forward_args1(self) -> None:
        self.output_assert(
            **self.basic_additional_forward_args_setup(),
            expected=torch.tensor([0]),  # attribution here is 0 so entropy is nan
        )

    def test_classification_convnet_multi_targets(self) -> None:
        self.output_assert(
            **self.classification_convnet_multi_targets_setup(),
            expected=torch.tensor([0.3840] * 20),
            delta=1.0e-3,
        )

    def test_classification_tpl_target(self) -> None:
        self.output_assert(
            **self.classification_tpl_target_setup(),
            expected=torch.tensor([0.2222, 0.0889, 0.0556, 0.0404]),
            delta=1.0e-3,
        )

    def test_classification_tpl_target_w_baseline(self) -> None:
        self.output_assert(
            **self.classification_tpl_target_w_baseline_setup(),
            expected=torch.tensor([0.4444, 0.1111, 0.0635, 0.0444]),
            delta=1.0e-3,
        )

    def output_assert(
        self,
        explainer: Union[FusionExplainer, Attribution],
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        expected: Tensor,
        additional_forward_args: Optional[Any] = None,
        baselines: Optional[BaselineType] = None,
        target: Optional[TargetType] = None,
        multiply_by_inputs: bool = False,
        delta: float = 1e-4,
    ) -> Tensor:
        explanations = self.compute_explanations(
            explainer,
            inputs,
            additional_forward_args,
            baselines,
            target,
            multiply_by_inputs,
        )
        output = sparseness(
            attributions=explanations,
        )
        if torch.isnan(output).all():
            assert torch.isnan(expected).all()
        else:
            assertTensorAlmostEqual(self, output, expected, delta=delta)
        return output


if __name__ == "__main__":
    unittest.main()
