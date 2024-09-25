import logging
import unittest
from logging import getLogger
from typing import Any, Optional, Union

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
from torchxai.explanation_framework.explainers.factory import ExplainerFactory
from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.metrics._utils.common import _tuple_tensors_to_tensors
from torchxai.metrics.faithfulness.monotonicity import monotonicity

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class Test(MetricTestsBase):
    def test_park_function(self) -> None:
        kwargs = self.park_function_setup()
        model = kwargs["model"]
        kwargs.pop("explainer")
        for explainer, expected in zip(
            [
                "random",
                "saliency",
                "input_x_gradient",
                "integrated_gradients",
            ],
            [False, True, True, True],
        ):
            self.output_assert(
                **kwargs,
                explainer=ExplainerFactory.create(explainer, model),
                expected=torch.tensor([expected]),
            )

    def test_basic_single(self) -> None:
        monotonicity_per_run = []
        fwds_per_run = []
        for max_features_processed_per_batch in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.output_assert(
                **self.basic_single_setup(),
                expected=torch.tensor([True]),
                max_features_processed_per_batch=max_features_processed_per_batch,
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
        for max_features_processed_per_batch in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.output_assert(
                **self.basic_batch_setup(),
                expected=torch.tensor([True] * 3),
                max_features_processed_per_batch=max_features_processed_per_batch,
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
        for max_features_processed_per_batch in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.output_assert(
                **self.basic_additional_forward_args_setup(),
                expected=torch.tensor([False]),
                max_features_processed_per_batch=max_features_processed_per_batch,
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
        for max_features_processed_per_batch in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.output_assert(
                **self.classification_convnet_multi_targets_setup(),
                expected=torch.tensor(
                    [True] * 20,
                    dtype=torch.float64,
                ),
                max_features_processed_per_batch=max_features_processed_per_batch,
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
        for max_features_processed_per_batch in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.output_assert(
                **self.classification_tpl_target_setup(),
                expected=torch.tensor([True] * 4),
                max_features_processed_per_batch=max_features_processed_per_batch,
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
        for max_features_processed_per_batch in [
            1,
            None,
            40,
        ]:
            mono, fwds = self.output_assert(
                **self.classification_tpl_target_w_baseline_setup(),
                expected=torch.tensor([True] * 4),
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            monotonicity_per_run.append(mono)
            fwds_per_run.append(fwds)
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in monotonicity_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(self, fwds_per_run)

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
        max_features_processed_per_batch: int = None,
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
        monotonicity_result, fwds = monotonicity(
            forward_func=model,
            inputs=inputs,
            attributions=explanations,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            target=target,
            max_features_processed_per_batch=max_features_processed_per_batch,
        )
        explanations, _ = _tuple_tensors_to_tensors(explanations)
        self.assertEqual(len(fwds), explanations.shape[0])  # match batch size
        for fwd, attribution in zip(fwds, explanations):
            self.assertEqual(
                fwd.numel(), attribution.numel()
            )  # match number of features
        assertTensorAlmostEqual(self, monotonicity_result.float(), expected.float())
        return monotonicity_result, fwds


if __name__ == "__main__":
    unittest.main()
