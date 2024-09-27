# import logging
# import unittest
# from logging import getLogger

# import torch

# from tests.explainers..explainers.base import ExplainersTestBase
# from torchxai.explainers.utils.common import grid_segmenter

# logging.basicConfig(level=logging.INFO)
# logger = getLogger(__name__)

# from torchxai.explainers._perturbation.feature_ablation import (
#     FeatureAblationExplainer,
# )


# class FeatureAblationExplainerTest(ExplainersTestBase):
#     def classification_alexnet_model_setup(self):
#         kwargs = super().classification_alexnet_model_setup()
#         kwargs["feature_mask"] = grid_segmenter(kwargs["inputs"], cell_size=32)
#         return kwargs

#     def test_basic_single(self) -> None:
#         self.basic_single_test_setup(
#             explainer_class=FeatureAblationExplainer,
#             expected_explanation=(torch.tensor([1.0]), torch.tensor([-1.0])),
#         )

#     def test_basic_single_batched(self) -> None:
#         self.basic_single_batched_test_setup(
#             explainer_class=FeatureAblationExplainer,
#             expected_explanation=(torch.tensor([[1.0]]), torch.tensor([[-1.0]])),
#         )

#     def test_basic_batched(self) -> None:
#         self.basic_batched_test_setup(
#             explainer_class=FeatureAblationExplainer,
#             expected_explanation=(
#                 torch.tensor([1, 1, 1]),
#                 torch.tensor([-1, -1, -1]),
#             ),
#         )

#     def test_basic_additional_forward_args(self) -> None:
#         self.basic_additional_forward_args_test_setup(
#             explainer_class=FeatureAblationExplainer,
#             expected_explanation=(
#                 torch.tensor([[0] * 3]),
#                 torch.tensor([[-0.5000, 0, 0]]),
#             ),
#         )

#     def test_classification_convnet_multi_target(self) -> None:
#         expected_explanation = (
#             torch.tensor(
#                 [
#                     [1.0, 4.0, 6.0, 4.0],
#                     [4.0, 10.0, 11.0, 8.0],
#                     [4.0, 14.0, 15.0, 12.0],
#                     [0.0, 0.0, 0.0, 0.0],
#                 ]
#             )
#             .unsqueeze(0)
#             .repeat(20, 1, 1, 1)
#         )
#         self.classification_convnet_multi_target_test_setup(
#             explainer_class=FeatureAblationExplainer,
#             expected_explanation=expected_explanation,
#         )

#     def test_classification_tpl_target_with_single_and_multiple_target_tests(
#         self,
#     ) -> None:
#         expected_explanations = [
#             (
#                 torch.tensor(
#                     [
#                         [32.0, 64.0, 88.0],
#                         [128.0, 160.0, 192.0],
#                         [224.0, 256.0, 288.0],
#                         [320.0, 352.0, 384.0],
#                     ]
#                 ),
#             ),
#             (
#                 torch.tensor(
#                     [
#                         [4.0, 8.0, 11.0],
#                         [128.0, 160.0, 192.0],
#                         [224.0, 256.0, 288.0],
#                         [320.0, 352.0, 384.0],
#                     ]
#                 ),
#             ),
#         ]
#         self.classification_tpl_target_test_setup_with_single_and_multiple_target_tests(
#             explainer_class=FeatureAblationExplainer,
#             expected_explanations=expected_explanations,
#         )

#     def test_classification_sigmoid_model_with_single_and_multiple_target_tests(
#         self,
#     ) -> None:
#         expected_explanations = [
#             (
#                 torch.tensor(
#                     [
#                         [
#                             0.0049,
#                             0.0232,
#                             0.0067,
#                             -0.0235,
#                             0.0079,
#                             -0.0225,
#                             0.0075,
#                             -0.0078,
#                             -0.0139,
#                             -0.0445,
#                         ]
#                     ]
#                 ),
#             ),
#             (
#                 torch.tensor(
#                     [
#                         [
#                             0.0093,
#                             0.0065,
#                             -0.0088,
#                             -0.0175,
#                             -0.0015,
#                             0.0075,
#                             0.0159,
#                             0.0035,
#                             -0.0197,
#                             0.0292,
#                         ]
#                     ]
#                 ),
#             ),
#         ]
#         self.classification_sigmoid_model_test_setup_with_single_and_multiple_target_tests(
#             explainer_class=FeatureAblationExplainer,
#             expected_explanations=expected_explanations,
#         )

#     def test_classification_softmax_model_with_single_and_multiple_target_tests(
#         self,
#     ) -> None:
#         expected_explanations = [
#             (
#                 torch.tensor(
#                     [
#                         [
#                             -7.0127e-04,
#                             4.7706e-04,
#                             -1.5219e-03,
#                             8.3491e-05,
#                             -4.4073e-04,
#                             -8.4874e-04,
#                             6.9338e-04,
#                             8.9946e-04,
#                             1.8473e-03,
#                             -2.4041e-04,
#                         ],
#                         [
#                             -3.7812e-04,
#                             5.1151e-03,
#                             -2.7881e-03,
#                             3.0873e-03,
#                             1.3787e-04,
#                             -2.8097e-03,
#                             1.3350e-03,
#                             2.3398e-03,
#                             2.1841e-03,
#                             -2.3115e-03,
#                         ],
#                         [
#                             -6.7709e-04,
#                             6.8232e-03,
#                             -4.5188e-03,
#                             3.7991e-03,
#                             -2.8683e-04,
#                             -7.2636e-03,
#                             1.0653e-04,
#                             3.0916e-03,
#                             2.7262e-03,
#                             -6.6334e-03,
#                         ],
#                     ]
#                 ),
#             ),
#             (
#                 torch.tensor(
#                     [
#                         [
#                             5.0232e-03,
#                             1.1835e-04,
#                             1.5855e-03,
#                             4.4607e-05,
#                             1.4812e-03,
#                             5.2779e-03,
#                             -1.8666e-03,
#                             -2.3631e-03,
#                             -5.4629e-03,
#                             2.5512e-03,
#                         ],
#                         [
#                             1.1770e-02,
#                             5.3996e-03,
#                             3.6647e-03,
#                             4.2853e-03,
#                             2.7804e-03,
#                             9.3164e-03,
#                             -3.5614e-03,
#                             -2.3597e-03,
#                             -1.1550e-02,
#                             4.2572e-03,
#                         ],
#                         [
#                             1.9662e-02,
#                             8.3726e-03,
#                             5.3482e-03,
#                             6.6263e-03,
#                             3.4180e-03,
#                             1.1877e-02,
#                             -7.9942e-03,
#                             -3.0180e-03,
#                             -1.8526e-02,
#                             3.4027e-03,
#                         ],
#                     ]
#                 ),
#             ),
#         ]
#         self.classification_softmax_model_test_setup_with_single_and_multiple_target_tests(
#             explainer_class=FeatureAblationExplainer,
#             expected_explanations=expected_explanations,
#         )

#     def test_classification_alexnet_model_with_single_and_multiple_targets_1(
#         self,
#     ) -> None:
#         self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_1(
#             explainer_class=FeatureAblationExplainer,
#         )

#     def test_classification_alexnet_model_with_single_and_multiple_targets_2(
#         self,
#     ) -> None:
#         self.classification_alexnet_model_test_setup_with_single_and_multiple_targets_2(
#             explainer_class=FeatureAblationExplainer,
#         )


# if __name__ == "__main__":
#     unittest.main()
