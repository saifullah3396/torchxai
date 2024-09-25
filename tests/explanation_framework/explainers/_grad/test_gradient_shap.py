import logging
import unittest
from logging import getLogger

import torch

from tests.explanation_framework.explainers.base import ExplainersTestBase

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

from torchxai.explanation_framework.explainers._grad.gradient_shap import (
    GradientShapExplainer,
)


class GradientShapExplainerTest(ExplainersTestBase):
    def basic_single_setup(self) -> None:
        kwargs = super().basic_single_setup()
        kwargs["baselines"] = (torch.randn(20), torch.randn(20))
        return kwargs

    def basic_single_batched_setup(self) -> None:
        kwargs = super().basic_single_batched_setup()
        kwargs["baselines"] = (
            torch.randn(20).unsqueeze(1),
            torch.randn(20).unsqueeze(1),
        )
        return kwargs

    def basic_batch_setup(self) -> None:
        kwargs = super().basic_batch_setup()
        kwargs["baselines"] = (torch.randn(20), torch.randn(20))
        return kwargs

    def basic_additional_forward_args_setup(self):
        kwargs = super().basic_additional_forward_args_setup()
        kwargs["baselines"] = (
            torch.randn(20, 3),
            torch.randn(20, 3),
        )
        return kwargs

    def classification_convnet_multi_target_setup(self):
        kwargs = super().classification_convnet_multi_target_setup()
        kwargs["baselines"] = torch.randn_like(kwargs["inputs"][0].repeat(20, 1, 1, 1))
        return kwargs

    def classification_tpl_target_setup(self):
        kwargs = super().classification_tpl_target_setup()
        kwargs["baselines"] = torch.randn_like(kwargs["inputs"])
        return kwargs

    def classification_sigmoid_model_setup(self):
        kwargs = super().classification_sigmoid_model_setup()
        kwargs["baselines"] = torch.randn(20, 10)
        return kwargs

    def classification_softmax_model_setup(self):
        kwargs = super().classification_softmax_model_setup()
        kwargs["baselines"] = torch.randn(20, 10)
        return kwargs

    def classification_alexnet_model_setup(self):
        kwargs = super().classification_alexnet_model_setup()
        kwargs["baselines"] = torch.randn_like(kwargs["inputs"])
        return kwargs

    def test_basic_single(self) -> None:
        self.basic_single_test_setup(
            explainer_class=GradientShapExplainer,
            expected_explanation=(torch.tensor([1.3201]), torch.tensor([-0.4517])),
            delta=1e-3,
        )

    def test_basic_single_batched(self) -> None:
        self.basic_single_batched_test_setup(
            explainer_class=GradientShapExplainer,
            expected_explanation=(torch.tensor([[1.3201]]), torch.tensor([[-0.4517]])),
            delta=1e-3,
        )

    def test_basic_batched(self) -> None:
        self.basic_batched_test_setup(
            explainer_class=GradientShapExplainer,
            expected_explanation=(
                torch.tensor([1.5404, 0.9260, 1.3761]),
                torch.tensor([-0.4602, -0.3535, -0.4157]),
            ),
            delta=1e-3,
        )

    def test_basic_additional_forward_args(self) -> None:
        self.basic_additional_forward_args_test_setup(
            explainer_class=GradientShapExplainer,
            expected_explanation=(
                torch.tensor([[0.0179, 0, 0]]),
                torch.tensor([[0] * 3]),
            ),
            delta=1e-3,
        )

    def test_classification_convnet_multi_target(self) -> None:
        self.classification_convnet_multi_target_test_setup(
            explainer_class=GradientShapExplainer,
            expected_explanation=None,
        )

    def test_classification_tpl_target_with_single_and_multiple_target_tests(
        self,
    ) -> None:
        expected_explanations = [
            (
                torch.tensor(
                    [
                        [37.1057, 32.8007, 85.5983],
                        [140.3305, 134.4731, 191.7730],
                        [228.3884, 237.6463, 275.9260],
                        [329.5155, 330.7893, 386.4172],
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [4.6382, 4.1001, 10.6998],
                        [140.3305, 134.4731, 191.7730],
                        [228.3884, 237.6463, 275.9260],
                        [329.5155, 330.7893, 386.4172],
                    ]
                ),
            ),
        ]
        self.classification_tpl_target_test_setup_with_single_and_multiple_target_tests(
            explainer_class=GradientShapExplainer,
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
                            0.0064,
                            0.0384,
                            0.0109,
                            -0.0108,
                            0.0126,
                            -0.0302,
                            0.0078,
                            -0.0039,
                            -0.0025,
                            -0.0500,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            0.0025,
                            0.0016,
                            -0.0064,
                            -0.0226,
                            -0.0077,
                            0.0077,
                            0.0044,
                            -0.0053,
                            -0.0230,
                            0.0343,
                        ]
                    ]
                ),
            ),
        ]
        self.classification_sigmoid_model_test_setup_with_single_and_multiple_target_tests(
            explainer_class=GradientShapExplainer,
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
                            -1.6180e-03,
                            -6.9099e-04,
                            -2.9489e-03,
                            5.5984e-04,
                            6.8739e-04,
                            -8.7574e-05,
                            2.7123e-04,
                            -6.5949e-04,
                            1.0477e-03,
                            -7.8138e-04,
                        ]
                    ]
                ),
            ),
            (
                torch.tensor(
                    [
                        [
                            4.4049e-03,
                            8.6631e-05,
                            2.6887e-03,
                            -8.3813e-04,
                            2.7784e-04,
                            4.9466e-03,
                            -1.0878e-03,
                            -8.4017e-04,
                            -5.0257e-03,
                            3.2244e-03,
                        ]
                    ]
                ),
            ),
        ]
        self.classification_softmax_model_test_setup_with_single_and_multiple_target_tests(
            explainer_class=GradientShapExplainer,
            expected_explanations=expected_explanations,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_1(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_1(
            explainer_class=GradientShapExplainer,
            internal_batch_size=64,
        )

    def test_classification_alexnet_model_with_single_and_multiple_targets_2(
        self,
    ) -> None:
        self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_2(
            explainer_class=GradientShapExplainer,
            internal_batch_size=64,
        )


if __name__ == "__main__":
    unittest.main()
