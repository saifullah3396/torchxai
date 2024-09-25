import logging
import unittest
from logging import getLogger

import torch

from tests.explanation_framework.explainers.base import ExplainersTestBase

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

from torchxai.explanation_framework.explainers._grad.deeplift import DeepLiftExplainer


class DeepLiftExplainerTest(ExplainersTestBase):
    def test_basic_single(self) -> None:
        self.basic_single_test_setup(
            explainer_class=DeepLiftExplainer,
            expected_explanation=(torch.tensor([3]), torch.tensor([-1])),
        )

    def test_basic_single_batched(self) -> None:
        self.basic_single_batched_test_setup(
            explainer_class=DeepLiftExplainer,
            expected_explanation=(torch.tensor([[3]]), torch.tensor([[-1]])),
        )

    def test_basic_batched(self) -> None:
        self.basic_batched_test_setup(
            explainer_class=DeepLiftExplainer,
            expected_explanation=(
                torch.tensor([3, 3, 3]),
                torch.tensor([-1, -1, -1]),
            ),
        )

    def test_basic_additional_forward_args(self) -> None:
        self.basic_additional_forward_args_test_setup(
            explainer_class=DeepLiftExplainer,
            expected_explanation=(torch.tensor([[0] * 3]), torch.tensor([[0] * 3])),
        )

    def test_classification_convnet_multi_target(self) -> None:
        expected_explanation = (
            torch.tensor(
                [
                    [
                        [0.0741, 0.5608, 0.8413, 0.8254],
                        [1.7593, 4.8644, 5.6751, 3.6710],
                        [3.1667, 8.1073, 8.9180, 5.5065],
                        [3.6111, 7.4242, 7.9545, 4.0404],
                    ]
                ]
            )
            .unsqueeze(0)
            .repeat(20, 1, 1, 1)
        )
        self.classification_convnet_multi_target_test_setup(
            explainer_class=DeepLiftExplainer,
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
                        [26.6667, 53.3333, 80.0000],
                        [128.0000, 160.0000, 192.0000],
                        [224.0000, 256.0000, 288.0000],
                        [320.0000, 352.0000, 384.0000],
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [3.3333, 6.6667, 10.0000],
                        [128.0000, 160.0000, 192.0000],
                        [224.0000, 256.0000, 288.0000],
                        [320.0000, 352.0000, 384.0000],
                    ]
                ),
            ),
        ]
        self.classification_tpl_target_test_setup_with_single_and_multiple_target_tests(
            explainer_class=DeepLiftExplainer,
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
            explainer_class=DeepLiftExplainer,
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
                            -1.3351e-03,
                            2.0642e-04,
                            -1.9536e-03,
                            -2.3064e-05,
                            -1.6667e-04,
                            -1.0324e-03,
                            -5.3702e-04,
                            4.5644e-05,
                            -1.2087e-04,
                            2.0013e-04,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            3.8315e-03,
                            9.0658e-04,
                            6.9715e-04,
                            5.0991e-05,
                            1.6153e-03,
                            4.4276e-03,
                            3.0104e-05,
                            -1.3260e-03,
                            -3.4423e-03,
                            2.1783e-03,
                        ]
                    ]
                ),
            ),
        ]
        self.classification_softmax_model_test_setup_with_single_and_multiple_target_tests(
            explainer_class=DeepLiftExplainer,
            expected_explanations=expected_explanations,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_1(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_1(
            explainer_class=DeepLiftExplainer,
            internal_batch_size=64,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_2(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_2(
            explainer_class=DeepLiftExplainer,
            internal_batch_size=64,
        )


if __name__ == "__main__":
    unittest.main()
