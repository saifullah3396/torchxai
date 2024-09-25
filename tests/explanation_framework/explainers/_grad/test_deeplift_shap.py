import logging
import unittest
from logging import getLogger

import torch

from tests.explanation_framework.explainers.base import ExplainersTestBase

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

from torchxai.explanation_framework.explainers._grad.deeplift_shap import (
    DeepLiftShapExplainer,
)


class DeepLiftShapExplainerTest(ExplainersTestBase):
    def basic_single_setup(self) -> None:
        kwargs = super().basic_single_setup()
        kwargs["train_baselines"] = (torch.randn(20), torch.randn(20))
        return kwargs

    def basic_single_batched_setup(self) -> None:
        kwargs = super().basic_single_batched_setup()
        kwargs["train_baselines"] = (
            torch.randn(20).unsqueeze(1),
            torch.randn(20).unsqueeze(1),
        )
        return kwargs

    def basic_batch_setup(self) -> None:
        kwargs = super().basic_batch_setup()
        kwargs["train_baselines"] = (torch.randn(20), torch.randn(20))
        return kwargs

    def basic_additional_forward_args_setup(self):
        kwargs = super().basic_additional_forward_args_setup()
        kwargs["train_baselines"] = (
            torch.randn(20, 3),
            torch.randn(20, 3),
        )
        return kwargs

    def classification_convnet_multi_target_setup(self):
        kwargs = super().classification_convnet_multi_target_setup()
        kwargs["train_baselines"] = torch.randn_like(
            kwargs["inputs"][0].repeat(20, 1, 1, 1)
        )
        return kwargs

    def classification_tpl_target_setup(self):
        kwargs = super().classification_tpl_target_setup()
        kwargs["train_baselines"] = torch.randn_like(kwargs["inputs"])
        return kwargs

    def classification_sigmoid_model_setup(self):
        kwargs = super().classification_sigmoid_model_setup()
        kwargs["train_baselines"] = torch.randn(20, 10)
        return kwargs

    def classification_softmax_model_setup(self):
        kwargs = super().classification_softmax_model_setup()
        kwargs["train_baselines"] = torch.randn(20, 10)
        return kwargs

    def classification_alexnet_model_setup(self):
        kwargs = super().classification_alexnet_model_setup()
        kwargs["train_baselines"] = torch.randn_like(kwargs["inputs"])
        return kwargs

    def test_basic_single(self) -> None:
        self.basic_single_test_setup(
            explainer_class=DeepLiftShapExplainer,
            expected_explanation=(torch.tensor([3.2823]), torch.tensor([-1.1275])),
        )

    def test_basic_single_batched(self) -> None:
        self.basic_single_batched_test_setup(
            explainer_class=DeepLiftShapExplainer,
            expected_explanation=(torch.tensor([[3.2823]]), torch.tensor([[-1.1275]])),
            delta=1e-4,
        )

    def test_basic_batched(self) -> None:
        self.basic_batched_test_setup(
            explainer_class=DeepLiftShapExplainer,
            expected_explanation=(
                torch.tensor([3.2823, 3.2823, 3.2823]),
                torch.tensor([-1.1275, -1.1275, -1.1275]),
            ),
            delta=1e-4,
        )

    def test_basic_additional_forward_args(self) -> None:
        self.basic_additional_forward_args_test_setup(
            explainer_class=DeepLiftShapExplainer,
            expected_explanation=(torch.tensor([[0] * 3]), torch.tensor([[0] * 3])),
        )

    def test_classification_convnet_multi_target(self) -> None:
        expected_explanation = (
            torch.tensor(
                [
                    [
                        [0.0927, 0.5919, 0.7778, 0.8637],
                        [1.7825, 4.8079, 5.9289, 3.6591],
                        [3.1948, 8.0690, 9.0044, 5.4237],
                        [3.6379, 7.3576, 7.7761, 4.0319],
                    ]
                ]
            )
            .unsqueeze(0)
            .repeat(20, 1, 1, 1)
        )
        self.classification_convnet_multi_target_test_setup(
            explainer_class=DeepLiftShapExplainer,
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
                        [34.5385, 36.5562, 77.4178],
                        [137.2962, 138.6327, 188.7545],
                        [233.2962, 234.6327, 284.7545],
                        [329.2962, 330.6327, 380.7545],
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [4.3173, 4.5695, 9.6772],
                        [137.2962, 138.6327, 188.7545],
                        [233.2962, 234.6327, 284.7545],
                        [329.2962, 330.6327, 380.7545],
                    ]
                ),
            ),
        ]
        self.classification_tpl_target_test_setup_with_single_and_multiple_target_tests(
            explainer_class=DeepLiftShapExplainer,
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
                            0.0121,
                            0.0323,
                            0.0076,
                            -0.0103,
                            0.0134,
                            -0.0246,
                            0.0078,
                            -0.0021,
                            -0.0052,
                            -0.0457,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0016,
                            0.0025,
                            -0.0060,
                            -0.0198,
                            -0.0048,
                            0.0100,
                            0.0052,
                            -0.0043,
                            -0.0185,
                            0.0302,
                        ]
                    ],
                ),
            ),
        ]
        self.classification_sigmoid_model_test_setup_with_single_and_multiple_target_tests(
            explainer_class=DeepLiftShapExplainer,
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
                            4.6330e-04,
                            -1.4664e-03,
                            -1.5381e-03,
                            4.7628e-05,
                            -2.2228e-04,
                            6.3917e-04,
                            2.8912e-04,
                            -2.4357e-04,
                            -7.1498e-04,
                            -1.1211e-03,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0036,
                            -0.0002,
                            0.0021,
                            0.0002,
                            0.0001,
                            0.0025,
                            -0.0011,
                            -0.0014,
                            -0.0035,
                            0.0012,
                        ]
                    ]
                ),
            ),
        ]
        self.classification_softmax_model_test_setup_with_single_and_multiple_target_tests(
            explainer_class=DeepLiftShapExplainer,
            expected_explanations=expected_explanations,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_1(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_1(
            explainer_class=DeepLiftShapExplainer,
            internal_batch_size=64,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_2(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_2(
            explainer_class=DeepLiftShapExplainer,
            internal_batch_size=64,
        )


if __name__ == "__main__":
    unittest.main()
