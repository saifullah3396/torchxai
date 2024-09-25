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
    set_all_random_seeds,
)
from tests.metrics.base import MetricTestsBase
from torchxai.explanation_framework.explainers.factory import ExplainerFactory
from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.metrics._utils.perturbation import default_random_perturb_func
from torchxai.metrics.complexity.effective_complexity import effective_complexity

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
                3,
                3,
                4,
            ],  # these effective complexity results match from the paper: https://arxiv.org/pdf/2007.07584
        ):
            self.output_assert(
                **kwargs,
                explainer=ExplainerFactory.create(explainer, model),
                perturb_func=default_random_perturb_func(noise_scale=1.0),
                expected=torch.tensor([expected]),
                n_perturbations_per_feature=10,
                eps=1e-2,
            )

    def test_basic_single(self) -> None:
        complexity_scores_list = []
        n_features_list = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            complexity_scores, _, n_features = self.output_assert(
                **self.basic_single_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([2]),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            complexity_scores_list.append(complexity_scores)
            n_features_list.append(n_features)

        assertAllTensorsAreAlmostEqualWithNan(self, complexity_scores_list)
        assertAllTensorsAreAlmostEqualWithNan(self, n_features_list)

    def test_basic_batch(self) -> None:
        complexity_scores_list = []
        n_features_list = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            complexity_scores, _, n_features = self.output_assert(
                **self.basic_batch_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([2, 2, 2]),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            complexity_scores_list.append(complexity_scores)
            n_features_list.append(n_features)

        assertAllTensorsAreAlmostEqualWithNan(self, complexity_scores_list)
        assertAllTensorsAreAlmostEqualWithNan(self, n_features_list)

    def test_basic_additional_forward_args1(self) -> None:
        complexity_scores_list = []
        n_features_list = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            complexity_scores, _, n_features = self.output_assert(
                **self.basic_additional_forward_args_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.zeros(1),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            complexity_scores_list.append(complexity_scores)
            n_features_list.append(n_features)

        assertAllTensorsAreAlmostEqualWithNan(self, complexity_scores_list)
        assertAllTensorsAreAlmostEqualWithNan(self, n_features_list)

    def test_classification_convnet_multi_targets(self) -> None:
        complexity_scores_list = []
        n_features_list = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            complexity_scores, _, n_features = self.output_assert(
                **self.classification_convnet_multi_targets_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([16] * 20),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            complexity_scores_list.append(complexity_scores)
            n_features_list.append(n_features)

        assertAllTensorsAreAlmostEqualWithNan(self, complexity_scores_list)
        assertAllTensorsAreAlmostEqualWithNan(self, n_features_list)

    def test_classification_convnet_multi_targets_eps_1(self) -> None:
        complexity_scores_list = []
        n_features_list = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            complexity_scores, _, n_features = self.output_assert(
                **self.classification_convnet_multi_targets_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([14] * 20),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
                eps=1e-2,
            )
            complexity_scores_list.append(complexity_scores)
            n_features_list.append(n_features)

        assertAllTensorsAreAlmostEqualWithNan(self, complexity_scores_list)
        assertAllTensorsAreAlmostEqualWithNan(self, n_features_list)

    def test_classification_convnet_multi_targets_eps_2(self) -> None:
        complexity_scores_list = []
        n_features_list = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            complexity_scores, _, n_features = self.output_assert(
                **self.classification_convnet_multi_targets_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([10] * 20),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
                eps=1e-1,
            )
            complexity_scores_list.append(complexity_scores)
            n_features_list.append(n_features)

        assertAllTensorsAreAlmostEqualWithNan(self, complexity_scores_list)
        assertAllTensorsAreAlmostEqualWithNan(self, n_features_list)

    def test_classification_tpl_target_eps_low(self) -> None:
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
                expected=torch.tensor([3, 3, 3, 3]),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
            )
            outputs.append(output)

        assertAllTensorsAreAlmostEqualWithNan(self, outputs)

    def test_classification_tpl_target_eps_high(self) -> None:
        outputs = []
        for n_perturbations_per_feature, max_features_processed_per_batch in zip(
            [10, 10, 20],
            [
                None,
                1,
                40,
            ],
        ):
            set_all_random_seeds(1234)
            output = self.output_assert(
                **self.classification_tpl_target_setup(),
                perturb_func=default_random_perturb_func(),
                expected=torch.tensor([1, 2, 2, 2]),
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
                eps=1e-1,
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
                expected=torch.tensor([3, 3, 3, 3]),
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
        feature_mask: TensorOrTupleOfTensorsGeneric = None,
        baselines: BaselineType = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        perturb_func: Callable = default_random_perturb_func(),
        n_perturbations_per_feature: int = 10,
        max_features_processed_per_batch: int = None,
        multiply_by_inputs: bool = False,
        eps: float = 1e-5,
        use_absolute_attributions: bool = True,
    ) -> Tensor:
        explanations = self.compute_explanations(
            explainer,
            inputs,
            additional_forward_args,
            baselines,
            target,
            multiply_by_inputs,
        )
        effective_complexity_score, k_features_perturbed_fwd_diff_vars, n_features = (
            effective_complexity(
                forward_func=model,
                inputs=inputs,
                attributions=explanations,
                feature_mask=feature_mask,
                additional_forward_args=additional_forward_args,
                target=target,
                perturb_func=perturb_func,
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
                eps=eps,
                use_absolute_attributions=use_absolute_attributions,
            )
        )
        assertTensorAlmostEqual(self, effective_complexity_score, expected)
        return (
            effective_complexity_score,
            k_features_perturbed_fwd_diff_vars,
            n_features,
        )


if __name__ == "__main__":
    unittest.main()
