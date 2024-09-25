import logging
import unittest
from logging import getLogger

import torch

from tests.explanation_framework.explainers.base import ExplainersTestBase

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

from torchxai.explanation_framework.explainers._grad.guided_backprop import (
    GuidedBackpropExplainer,
)


class GuidedBackpropExplainerTest(ExplainersTestBase):
    def test_basic_single(self) -> None:
        self.basic_single_test_setup(
            explainer_class=GuidedBackpropExplainer,
            expected_explanation=(torch.tensor([1.0]), torch.tensor([-1.0])),
        )

    def test_basic_single_batched(self) -> None:
        self.basic_single_batched_test_setup(
            explainer_class=GuidedBackpropExplainer,
            expected_explanation=(torch.tensor([[1.0]]), torch.tensor([[-1.0]])),
        )

    def test_basic_batched(self) -> None:
        self.basic_batched_test_setup(
            explainer_class=GuidedBackpropExplainer,
            expected_explanation=(
                torch.tensor([1, 1, 1]),
                torch.tensor([-1, -1, -1]),
            ),
        )

    def test_basic_additional_forward_args(self) -> None:
        self.basic_additional_forward_args_test_setup(
            explainer_class=GuidedBackpropExplainer,
            expected_explanation=(torch.tensor([[0] * 3]), torch.tensor([[0] * 3])),
        )

    def test_classification_convnet_multi_target(self) -> None:
        expected_explanation = (
            torch.tensor([[1, 2, 2, 1], [2, 4, 4, 2], [2, 4, 4, 2], [1, 2, 2, 1]])
            .unsqueeze(0)
            .repeat(20, 1, 1, 1)
        )
        self.classification_convnet_multi_target_test_setup(
            explainer_class=GuidedBackpropExplainer,
            expected_explanation=expected_explanation,
        )

    def test_classification_tpl_target_with_single_and_multiple_target_tests(
        self,
    ) -> None:
        expected_explanations = [
            (torch.tensor([32]).expand(4, 3),),
            (
                torch.cat(
                    [torch.tensor(4).expand(1, 3), torch.tensor([32]).expand(3, 3)],
                    dim=0,
                ),
            ),
        ]
        self.classification_tpl_target_test_setup_with_single_and_multiple_target_tests(
            explainer_class=GuidedBackpropExplainer,
            expected_explanations=expected_explanations,
        )

    def test_classification_sigmoid_model_with_single_and_multiple_target_tests(
        self,
    ) -> None:
        expected_explanations = [
            (
                torch.tensor(
                    [
                        [
                            0.0192,
                            0.0109,
                            0.0129,
                            -0.0030,
                            0.0182,
                            0.0138,
                            0.0028,
                            -0.0147,
                            0.0061,
                            -0.0125,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0141,
                            0.0017,
                            0.0027,
                            -0.0140,
                            0.0056,
                            0.0257,
                            0.0139,
                            -0.0090,
                            -0.0061,
                            0.0107,
                        ]
                    ]
                ),
            ),
        ]
        self.classification_sigmoid_model_test_setup_with_single_and_multiple_target_tests(
            explainer_class=GuidedBackpropExplainer,
            expected_explanations=expected_explanations,
        )

    def test_classification_softmax_model_with_single_and_multiple_target_tests(
        self,
    ) -> None:
        expected_explanations = [
            (
                torch.tensor(
                    [
                        [
                            3.7760e-04,
                            8.5276e-05,
                            2.0446e-04,
                            1.6386e-04,
                            1.1548e-03,
                            1.5595e-03,
                            1.1076e-04,
                            -5.6630e-04,
                            1.8906e-04,
                            -2.2819e-05,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0046,
                            0.0001,
                            0.0025,
                            0.0015,
                            0.0028,
                            0.0061,
                            -0.0013,
                            -0.0024,
                            -0.0027,
                            0.0025,
                        ]
                    ]
                ),
            ),
        ]
        self.classification_softmax_model_test_setup_with_single_and_multiple_target_tests(
            explainer_class=GuidedBackpropExplainer,
            expected_explanations=expected_explanations,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_1(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_1(
            explainer_class=GuidedBackpropExplainer,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_2(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_2(
            explainer_class=GuidedBackpropExplainer,
        )


if __name__ == "__main__":
    unittest.main()
