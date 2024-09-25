import logging
import unittest
from logging import getLogger
from typing import Any, Callable, Optional, Union

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
from torchxai.metrics.axiomatic.monotonicity_corr_and_non_sens import (
    default_random_perturb_func,
    monotonicity_corr_and_non_sens,
)

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
                0,
                0,
                0,
            ],  # these non-sensitivity results match from the paper: https://arxiv.org/pdf/2007.07584
        ):
            self.output_assert(
                **kwargs,
                explainer=ExplainerFactory.create(explainer, model),
                perturb_func=default_random_perturb_func(noise_scale=1.0),
                expected=torch.tensor([expected]),
                n_perturbations_per_feature=100,
            )

    def test_basic_single(self) -> None:
        outputs = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            output = self.output_assert(
                **self.basic_single_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.zeros(1),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_basic_batch(self) -> None:
        outputs = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            output = self.output_assert(
                **self.basic_batch_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.zeros(3),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_basic_additional_forward_args1(self) -> None:
        outputs = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            output = self.output_assert(
                **self.basic_additional_forward_args_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.ones(1),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_classification_convnet_multi_targets(self) -> None:
        outputs = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            output = self.output_assert(
                **self.classification_convnet_multi_targets_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([4] * 20),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_classification_tpl_target(self) -> None:
        outputs = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            output = self.output_assert(
                **self.classification_tpl_target_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([0, 0, 0, 0]),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_classification_tpl_target_w_baseline(self) -> None:
        outputs = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            output = self.output_assert(
                **self.classification_tpl_target_w_baseline_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([1, 0, 0, 0]),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def output_assert(
        self,
        expected: Tensor,
        explainer: Union[Attribution, FusionExplainer],
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: Optional[Any] = None,
        baselines: BaselineType = None,
        feature_mask: TensorOrTupleOfTensorsGeneric = None,
        target: Optional[TargetType] = None,
        multiply_by_inputs: bool = False,
        perturb_func: Callable = default_random_perturb_func(),
        n_perturbations_per_feature: int = 10,
        max_features_processed_per_batch: int = None,
    ) -> Tensor:
        explanations = self.compute_explanations(
            explainer,
            inputs,
            additional_forward_args,
            baselines,
            target,
            multiply_by_inputs,
        )
        _, non_sens, _ = monotonicity_corr_and_non_sens(
            forward_func=model,
            inputs=inputs,
            attributions=explanations,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            target=target,
            perturb_func=perturb_func,
            n_perturbations_per_feature=n_perturbations_per_feature,
            max_features_processed_per_batch=max_features_processed_per_batch,
        )
        assertTensorAlmostEqual(self, non_sens, expected)
        return non_sens


if __name__ == "__main__":
    unittest.main()
