import logging
import unittest
from logging import getLogger

import torch

from tests.explanation_framework.explainers.base import ExplainersTestBase

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

from torchxai.explanation_framework.explainers._grad.input_x_gradient import (
    InputXGradientExplainer,
)


class InputXGradientExplainerTest(ExplainersTestBase):
    def test_basic_single(self) -> None:
        self.basic_single_test_setup(
            explainer_class=InputXGradientExplainer,
            expected_explanation=(torch.tensor([3.0]), torch.tensor([-1.0])),
        )

    def test_basic_single_batched(self) -> None:
        self.basic_single_batched_test_setup(
            explainer_class=InputXGradientExplainer,
            expected_explanation=(torch.tensor([[3.0]]), torch.tensor([[-1.0]])),
        )

    def test_basic_batched(self) -> None:
        self.basic_batched_test_setup(
            explainer_class=InputXGradientExplainer,
            expected_explanation=(
                torch.tensor([3, 3, 3]),
                torch.tensor([-1, -1, -1]),
            ),
        )

    def test_basic_additional_forward_args(self) -> None:
        self.basic_additional_forward_args_test_setup(
            explainer_class=InputXGradientExplainer,
            expected_explanation=(torch.tensor([[0] * 3]), torch.tensor([[0] * 3])),
        )

    def test_classification_convnet_multi_target(self) -> None:
        expected_explanation = (
            torch.tensor([[1, 4, 6, 4], [5, 12, 14, 8], [9, 20, 22, 12], [0, 0, 0, 0]])
            .unsqueeze(0)
            .repeat(20, 1, 1, 1)
        )
        self.classification_convnet_multi_target_test_setup(
            explainer_class=InputXGradientExplainer,
            expected_explanation=expected_explanation,
        )

    def test_classification_tpl_target_with_single_and_multiple_target_tests(
        self,
    ) -> None:
        expecetd_explanation_1 = torch.tensor(
            [
                [32.0, 64.0, 96.0],
                [128.0, 160.0, 192.0],
                [224.0, 256.0, 288.0],
                [320.0, 352.0, 384.0],
            ]
        )
        expecetd_explanation_2 = torch.tensor(
            [
                [4.0, 8.0, 12.0],
                [128.0, 160.0, 192.0],
                [224.0, 256.0, 288.0],
                [320.0, 352.0, 384.0],
            ]
        )
        expected_explanations = [
            expecetd_explanation_1,
            expecetd_explanation_2,
        ]
        self.classification_tpl_target_test_setup_with_single_and_multiple_target_tests(
            explainer_class=InputXGradientExplainer,
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
            explainer_class=InputXGradientExplainer,
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
            explainer_class=InputXGradientExplainer,
            expected_explanations=expected_explanations,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_1(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_1(
            explainer_class=InputXGradientExplainer,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_2(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_2(
            explainer_class=InputXGradientExplainer,
        )


if __name__ == "__main__":
    unittest.main()
