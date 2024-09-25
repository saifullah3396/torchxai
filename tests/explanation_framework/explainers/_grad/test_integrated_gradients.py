import logging
import unittest
from logging import getLogger

import torch

from tests.explanation_framework.explainers.base import ExplainersTestBase

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

from torchxai.explanation_framework.explainers._grad.integrated_gradients import (
    IntegratedGradientsExplainer,
)


class IntegratedGradientsExplainerTest(ExplainersTestBase):
    def test_basic_single(self) -> None:
        self.basic_single_test_setup(
            explainer_class=IntegratedGradientsExplainer,
            expected_explanation=(torch.tensor([1.5]), torch.tensor([-0.5])),
        )

    def test_basic_single_batched(self) -> None:
        self.basic_single_batched_test_setup(
            explainer_class=IntegratedGradientsExplainer,
            expected_explanation=(torch.tensor([[1.5]]), torch.tensor([[-0.5]])),
        )

    def test_basic_batched(self) -> None:
        self.basic_batched_test_setup(
            explainer_class=IntegratedGradientsExplainer,
            expected_explanation=(
                torch.tensor([1.5, 1.5, 1.5]),
                torch.tensor([-0.5, -0.5, -0.5]),
            ),
        )

    def test_basic_additional_forward_args(self) -> None:
        self.basic_additional_forward_args_test_setup(
            explainer_class=IntegratedGradientsExplainer,
            expected_explanation=(torch.tensor([[0] * 3]), torch.tensor([[0] * 3])),
        )

    def test_classification_convnet_multi_target(self) -> None:
        expected_explanation = (
            torch.tensor(
                [
                    [
                        [0.0743, 0.5622, 0.8432, 0.8270],
                        [1.7772, 4.8901, 5.7051, 3.6766],
                        [3.1990, 8.1501, 8.9651, 5.5149],
                        [3.6542, 7.4750, 8.0089, 4.0453],
                    ]
                ]
            )
            .unsqueeze(0)
            .repeat(20, 1, 1, 1)
        )
        self.classification_convnet_multi_target_test_setup(
            explainer_class=IntegratedGradientsExplainer,
            expected_explanation=expected_explanation,
            delta=1e-2,
        )

    def test_classification_tpl_target_with_single_and_multiple_target_tests(
        self,
    ) -> None:
        expected_explanations = [
            (
                torch.tensor(
                    [
                        [26.6483, 53.2966, 79.9448],
                        [128.0000, 160.0000, 192.0000],
                        [224.0000, 256.0000, 288.0000],
                        [320.0000, 352.0000, 384.0000],
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [3.3310, 6.6621, 9.9931],
                        [128.0000, 160.0000, 192.0000],
                        [224.0000, 256.0000, 288.0000],
                        [320.0000, 352.0000, 384.0000],
                    ]
                ),
            ),
        ]
        self.classification_tpl_target_test_setup_with_single_and_multiple_target_tests(
            explainer_class=IntegratedGradientsExplainer,
            expected_explanations=expected_explanations,
            delta=1e-3,
        )

    def test_classification_sigmoid_model_with_single_and_multiple_target_tests(
        self,
    ) -> None:
        expected_explanations = [
            (
                torch.tensor(
                    [
                        [
                            0.0145,
                            0.0263,
                            -0.0018,
                            -0.0162,
                            0.0174,
                            -0.0246,
                            0.0083,
                            -0.0043,
                            -0.0087,
                            -0.0404,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            -0.0034,
                            0.0028,
                            -0.0160,
                            -0.0191,
                            -0.0055,
                            0.0081,
                            0.0127,
                            -0.0014,
                            -0.0182,
                            0.0217,
                        ]
                    ],
                ),
            ),
        ]
        self.classification_sigmoid_model_test_setup_with_single_and_multiple_target_tests(
            explainer_class=IntegratedGradientsExplainer,
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
            explainer_class=IntegratedGradientsExplainer,
            expected_explanations=expected_explanations,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_1(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_1(
            explainer_class=IntegratedGradientsExplainer,
            internal_batch_size=64,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_2(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_2(
            explainer_class=IntegratedGradientsExplainer,
            internal_batch_size=64,
        )


if __name__ == "__main__":
    unittest.main()
