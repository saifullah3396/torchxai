import logging
import unittest
from logging import getLogger

import torch

from tests.explanation_framework.explainers.base import ExplainersTestBase

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

from torchxai.explanation_framework.explainers._grad.saliency import SaliencyExplainer


class SaliencyExplainerTest(ExplainersTestBase):
    def test_basic_single(self) -> None:
        self.basic_single_test_setup(
            explainer_class=SaliencyExplainer,
            expected_explanation=(torch.tensor([1.0]), torch.tensor([-1.0])),
        )

    def test_basic_single_batched(self) -> None:
        self.basic_single_batched_test_setup(
            explainer_class=SaliencyExplainer,
            expected_explanation=(torch.tensor([[1.0]]), torch.tensor([[-1.0]])),
        )

    def test_basic_batched(self) -> None:
        self.basic_batched_test_setup(
            explainer_class=SaliencyExplainer,
            expected_explanation=(
                torch.tensor([1, 1, 1]),
                torch.tensor([-1, -1, -1]),
            ),
        )

    def test_basic_additional_forward_args(self) -> None:
        self.basic_additional_forward_args_test_setup(
            explainer_class=SaliencyExplainer,
            expected_explanation=(torch.tensor([[0] * 3]), torch.tensor([[0] * 3])),
        )

    def test_classification_convnet_multi_target(self) -> None:
        expected_explanation = (
            torch.tensor([[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1], [0, 0, 0, 0]])
            .unsqueeze(0)
            .repeat(20, 1, 1, 1)
        )
        self.classification_convnet_multi_target_test_setup(
            explainer_class=SaliencyExplainer,
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
            explainer_class=SaliencyExplainer,
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
                            0.0157,
                            0.0233,
                            0.0016,
                            -0.0218,
                            0.0163,
                            -0.0256,
                            0.0075,
                            -0.0041,
                            -0.0143,
                            -0.0398,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            -0.0029,
                            0.0065,
                            -0.0146,
                            -0.0205,
                            -0.0104,
                            0.0038,
                            0.0160,
                            -0.0005,
                            -0.0218,
                            0.0233,
                        ]
                    ]
                ),
            ),
        ]
        self.classification_sigmoid_model_test_setup_with_single_and_multiple_target_tests(
            explainer_class=SaliencyExplainer,
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
                            -0.0025,
                            -0.0002,
                            -0.0024,
                            0.0006,
                            0.0010,
                            -0.0005,
                            0.0007,
                            0.0007,
                            0.0014,
                            -0.0027,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0047,
                            0.0001,
                            0.0021,
                            0.0016,
                            0.0024,
                            0.0064,
                            -0.0019,
                            -0.0027,
                            -0.0040,
                            0.0036,
                        ]
                    ]
                ),
            ),
        ]
        self.classification_softmax_model_test_setup_with_single_and_multiple_target_tests(
            explainer_class=SaliencyExplainer,
            expected_explanations=expected_explanations,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_1(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_1(
            explainer_class=SaliencyExplainer,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_2(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_2(
            explainer_class=SaliencyExplainer,
        )


if __name__ == "__main__":
    unittest.main()
