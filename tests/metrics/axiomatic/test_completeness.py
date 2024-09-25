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
from torchxai.metrics.axiomatic.completeness import completeness

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
                1.6322,  # saliency completeness is not so great
                0.1865,  # input_x_gradient results in better completeness
                1.3856e-08,  # integrated_gradients results in full completeness
            ],
        ):
            self.output_assert(
                **kwargs,
                explainer=ExplainerFactory.create(explainer, model),
                expected=torch.tensor([expected]),
            )

    def test_basic_single(self) -> None:
        self.output_assert(**self.basic_single_setup(), expected=torch.zeros(1))

    def test_basic_batch(self) -> None:
        self.output_assert(**self.basic_batch_setup(), expected=torch.zeros(3))

    def test_basic_additional_forward_args1(self) -> None:
        self.output_assert(
            **self.basic_additional_forward_args_setup(),
            expected=torch.zeros(1),
        )

    def test_classification_convnet_multi_targets(self) -> None:
        self.output_assert(
            **self.classification_convnet_multi_targets_setup(),
            expected=torch.zeros(20),
        )

    def test_classification_tpl_target(self) -> None:
        self.output_assert(
            **self.classification_tpl_target_setup(),
            expected=torch.tensor([0.6538, 0, 0, 0]),
        )

    def test_classification_tpl_target_w_baseline(self) -> None:
        self.output_assert(
            **self.classification_tpl_target_w_baseline_setup(),
            expected=torch.tensor([0.3269, 0, 0, 0]),
        )

    def output_assert(
        self,
        expected: Tensor,
        explainer: Union[Attribution, FusionExplainer],
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: Optional[Any] = None,
        baselines: Optional[BaselineType] = None,
        target: Optional[TargetType] = None,
        multiply_by_inputs: bool = False,
    ) -> Tensor:
        explanations = self.compute_explanations(
            explainer,
            inputs,
            additional_forward_args,
            baselines,
            target,
            multiply_by_inputs,
        )
        score = completeness(
            forward_func=model,
            inputs=inputs,
            attributions=explanations,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            target=target,
        )
        assertTensorAlmostEqual(self, score, expected)
        return score


if __name__ == "__main__":
    unittest.main()
