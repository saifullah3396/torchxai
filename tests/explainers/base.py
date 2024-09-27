import logging
from logging import getLogger
from typing import Any, List, Optional, Tuple, Type

import torch
import tqdm
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor
from torch.nn import Module

from tests.helpers.basic import (BaseTest, assertTensorAlmostEqual,
                                 set_all_random_seeds)
from tests.helpers.basic_models import (BasicModel2, BasicModel4MultiArgs,
                                        BasicModel_ConvNet_One_Conv,
                                        BasicModel_MultiLayer)
from tests.helpers.classification_models import SigmoidModel, SoftmaxModel
from torchxai.explainers.explainer import Explainer

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class ExplainersTestBase(BaseTest):
    def basic_single_setup(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([3.0])
        input2 = torch.tensor([1.0])
        inputs = (input1, input2)
        return dict(model=model, inputs=inputs)

    def basic_single_batched_setup(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[3.0]])
        input2 = torch.tensor([[1.0]])
        inputs = (input1, input2)
        return dict(model=model, inputs=inputs)

    def basic_batch_setup(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([3.0] * 3)
        input2 = torch.tensor([1.0] * 3)
        inputs = (input1, input2)
        return dict(model=model, inputs=inputs)

    def basic_additional_forward_args_setup(self):
        model = BasicModel4MultiArgs()
        input1 = torch.tensor([[1.5, 2.0, 3.3]])
        input2 = torch.tensor([[3.0, 3.5, 2.2]])
        inputs = (input1, input2)
        args = torch.tensor([[1.0, 3.0, 4.0]])
        return dict(
            model=model,
            inputs=inputs,
            additional_forward_args=(args,),
        )

    def classification_convnet_multi_target_setup(self):
        model = BasicModel_ConvNet_One_Conv()
        inputs = torch.stack([torch.arange(1, 17).float()] * 20, dim=0).view(
            20, 1, 4, 4
        )
        target = torch.tensor([1] * 20)
        return dict(model=model, inputs=inputs, target=target)

    def classification_tpl_target_setup(self):
        model = BasicModel_MultiLayer()
        inputs = torch.arange(1.0, 13.0).view(4, 3)
        additional_forward_args = (torch.arange(1, 13).view(4, 3).float(), True)
        target: List[Tuple[int, ...]] = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
        return dict(
            model=model,
            inputs=inputs,
            additional_forward_args=additional_forward_args,
            target=target,
        )

    def classification_tpl_target_w_baseline_setup(self):
        model = BasicModel_MultiLayer()
        inputs = torch.arange(1.0, 13.0).view(4, 3)
        baselines = torch.ones(4, 3)
        additional_forward_args = (torch.arange(1, 13).view(4, 3).float(), True)
        target: List[Tuple[int, ...]] = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
        return dict(
            model=model,
            inputs=inputs,
            additional_forward_args=additional_forward_args,
            target=target,
            baselines=baselines,
        )

    def classification_sigmoid_model_setup(self):
        model = SigmoidModel(10, 20, 10)
        inputs = torch.tensor([[1.0] * 10])
        return dict(model=model, inputs=inputs, target=torch.tensor([1]))

    def classification_softmax_model_setup(self):
        model = SoftmaxModel(10, 20, 10)
        inputs = torch.tensor([[1.0] * 10, [2.0] * 10, [3.0] * 10])
        return dict(model=model, inputs=inputs, target=torch.tensor([1]))

    def classification_alexnet_model_setup(self):
        from torchvision.models import alexnet

        model = alexnet(pretrained=False)
        model.eval()
        model.zero_grad()
        inputs = torch.randn(10, 3, 224, 224)
        return dict(model=model, inputs=inputs, target=torch.tensor([1]))

    def basic_single_test_setup(
        self,
        explainer_class: Type[Explainer],
        expected_explanation: Tuple[torch.Tensor, ...],
        delta: float = 1e-5,
    ) -> None:
        set_all_random_seeds(1234)
        self.basic_model_assert(
            **self.basic_single_setup(),
            explainer_class=explainer_class,
            target=None,
            expected_explanation=expected_explanation,
            delta=delta,
        )
        set_all_random_seeds(1234)
        self.basic_model_assert(
            **self.basic_single_setup(),
            explainer_class=explainer_class,
            target=0,
            expected_explanation=None,
            assert_failure=True,
            delta=delta,
        )
        set_all_random_seeds(1234)
        self.basic_model_assert(  # multi-target is not allowed when the model is not multi-target
            **self.basic_single_setup(),
            explainer_class=explainer_class,
            expected_explanation=[expected_explanation],
            target=None,
            is_multi_target=True,
            assert_failure=False,
            delta=delta,
        )
        set_all_random_seeds(1234)
        self.basic_model_assert(  # multi-target is not allowed when the model is not multi-target
            **self.basic_single_setup(),
            explainer_class=explainer_class,
            expected_explanation=None,
            target=0,
            is_multi_target=True,
            assert_failure=True,
            delta=delta,
        )

    def basic_single_batched_test_setup(
        self,
        explainer_class: Type[Explainer],
        expected_explanation: Tuple[torch.Tensor, ...],
        delta: float = 1e-5,
    ) -> None:
        set_all_random_seeds(1234)
        explanations = self.basic_model_assert(
            **self.basic_single_batched_setup(),
            target=0,
            explainer_class=explainer_class,
            expected_explanation=expected_explanation,
            delta=delta,
        )
        set_all_random_seeds(1234)
        multi_target_explanations = self.basic_model_assert(
            **self.basic_single_batched_setup(),
            explainer_class=explainer_class,
            expected_explanation=[expected_explanation],
            target=0,
            is_multi_target=True,
            delta=delta,
        )
        set_all_random_seeds(1234)
        multi_target_explanations = self.basic_model_assert(
            **self.basic_single_batched_setup(),
            explainer_class=explainer_class,
            expected_explanation=[expected_explanation],
            target=[0],
            is_multi_target=True,
            delta=delta,
        )

        self.compare_explanation_per_target(multi_target_explanations[0], explanations)

    def basic_batched_test_setup(
        self,
        explainer_class: Type[Explainer],
        expected_explanation: Tuple[torch.Tensor, ...],
        delta: float = 1e-5,
    ) -> None:
        set_all_random_seeds(1234)
        explanations = self.basic_model_assert(
            **self.basic_batch_setup(),
            explainer_class=explainer_class,
            expected_explanation=expected_explanation,
            delta=delta,
        )
        set_all_random_seeds(1234)
        multi_target_explanations = self.basic_model_assert(
            **self.basic_batch_setup(),
            explainer_class=explainer_class,
            expected_explanation=[expected_explanation],
            target=None,
            is_multi_target=True,
            delta=delta,
        )
        self.compare_explanation_per_target(multi_target_explanations[0], explanations)

    def basic_additional_forward_args_test_setup(
        self,
        explainer_class: Type[Explainer],
        expected_explanation: Tuple[torch.Tensor, ...],
        baseline_type: Optional[str] = None,
        delta: float = 1e-5,
    ) -> None:
        set_all_random_seeds(1234)
        explanations = self.basic_model_assert(
            **self.basic_additional_forward_args_setup(),
            explainer_class=explainer_class,
            expected_explanation=expected_explanation,
            delta=delta,
        )
        set_all_random_seeds(1234)
        multi_target_explanations = self.basic_model_assert(
            **self.basic_additional_forward_args_setup(),
            explainer_class=explainer_class,
            expected_explanation=[expected_explanation],
            target=None,
            is_multi_target=True,
            delta=delta,
        )
        self.compare_explanation_per_target(multi_target_explanations[0], explanations)

    def classification_convnet_multi_target_test_setup(
        self,
        explainer_class: Type[Explainer],
        expected_explanation: Tuple[torch.Tensor, ...],
        delta: float = 1e-5,
    ) -> None:
        set_all_random_seeds(1234)
        explanations = self.basic_model_assert(
            **self.classification_convnet_multi_target_setup(),
            explainer_class=explainer_class,
            expected_explanation=expected_explanation,
            delta=delta,
        )
        set_all_random_seeds(1234)
        multi_target_explanations = self.basic_model_assert(
            **self.classification_convnet_multi_target_setup(),
            explainer_class=explainer_class,
            expected_explanation=(
                [expected_explanation] if expected_explanation is not None else None
            ),
            is_multi_target=True,
            delta=delta,
        )
        self.compare_explanation_per_target(multi_target_explanations[0], explanations)

    def classification_tpl_target_test_setup_with_single_and_multiple_target_tests(
        self,
        explainer_class: Type[Explainer],
        expected_explanations: Tuple[torch.Tensor, ...],
        delta: float = 1e-5,
    ) -> None:
        self.single_multi_target_test_setup(
            base_setup_kwargs=self.classification_tpl_target_setup(),
            targets=[
                [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
                [(0, 0, 0), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
            ],
            explainer_class=explainer_class,
            expected_explanations=expected_explanations,
            delta=delta,
        )

    def classification_sigmoid_model_test_setup_with_single_and_multiple_target_tests(
        self,
        explainer_class: Type[Explainer],
        expected_explanations: Tuple[torch.Tensor, ...],
    ) -> None:
        self.single_multi_target_test_setup(
            base_setup_kwargs=self.classification_sigmoid_model_setup(),
            targets=[torch.tensor([0]), torch.tensor([1])],
            delta=1e-3,
            explainer_class=explainer_class,
            expected_explanations=expected_explanations,
        )

    def classification_softmax_model_test_setup_with_single_and_multiple_target_tests(
        self,
        explainer_class: Type[Explainer],
        expected_explanations: Tuple[torch.Tensor, ...],
    ) -> None:
        self.single_multi_target_test_setup(
            base_setup_kwargs=self.classification_softmax_model_setup(),
            targets=[torch.tensor([0]), torch.tensor([1])],
            delta=1e-3,
            explainer_class=explainer_class,
            expected_explanations=expected_explanations,
        )

    def classification_alexnet_model_test_setup_with_single_and_multiple_targets_1(
        self,
        explainer_class: Type[Explainer],
        internal_batch_size=16,
    ) -> None:
        self.single_multi_target_test_no_expected_output_setup(
            explainer_class=explainer_class,
            base_setup_kwargs=self.classification_alexnet_model_setup(),
            targets=[0, 1, 2],
            internal_batch_size=internal_batch_size,
        )

    def classification_alexnet_model_test_setup_with_single_and_multiple_targets_2(
        self,
        explainer_class: Type[Explainer],
        internal_batch_size=16,
    ) -> None:
        self.single_multi_target_test_no_expected_output_setup(
            explainer_class=explainer_class,
            base_setup_kwargs=self.classification_alexnet_model_setup(),
            targets=[
                [0] * 10,
                [1] * 10,
                list(range(10)),
            ],  # take all the outputs at 0th index as target
            internal_batch_size=internal_batch_size,
        )

    def single_multi_target_test_setup(
        self,
        explainer_class,
        base_setup_kwargs,
        targets,
        expected_explanations,
        delta=1e-5,
    ):
        # first test the case where the targets are sent as a list to multi-target and the same targets
        # are sent to captum based implementation as single target, the output should be same except that the
        # output should be a list of explanations in the case of multi-target
        kwargs = {**base_setup_kwargs}
        single_target_explanation_ouputs = []
        for target, expected_explanation in tqdm.tqdm(
            zip(targets, expected_explanations)
        ):
            kwargs["target"] = target
            set_all_random_seeds(1234)
            single_target_explanations = self.basic_model_assert(
                **kwargs,
                explainer_class=explainer_class,
                expected_explanation=expected_explanation,
                delta=delta,
            )
            kwargs["target"] = [target] if isinstance(target[0], tuple) else target
            set_all_random_seeds(1234)
            multi_target_explanations = self.basic_model_assert(
                **kwargs,
                explainer_class=explainer_class,
                expected_explanation=[expected_explanation],
                is_multi_target=True,
                delta=delta,
            )
            self.compare_explanation_per_target(
                multi_target_explanations[0],
                single_target_explanations,
            )
            single_target_explanation_ouputs.append(single_target_explanations)

        # now test the case where the targets are sent as a list to multi-target and the same targets and all
        # the explanations generated by our multi-target implementation should match the explanations generated
        # by the captum-based implementation
        kwargs["target"] = targets
        set_all_random_seeds(1234)
        multi_target_explanations = self.basic_model_assert(
            **kwargs,
            explainer_class=explainer_class,
            expected_explanation=expected_explanations,
            is_multi_target=True,
            delta=delta,
        )
        for multi_target_explanation, single_target_explanation in zip(
            multi_target_explanations, single_target_explanation_ouputs
        ):
            # target explanation in the list should match the single target explanations at the same index
            self.compare_explanation_per_target(
                multi_target_explanation,
                single_target_explanation,
            )

    def single_multi_target_test_no_expected_output_setup(
        self,
        explainer_class,
        base_setup_kwargs,
        targets,
        delta=1e-5,
        internal_batch_size=16,
    ):
        # first test the case where the targets are sent as a list to multi-target and the same targets
        # are sent to captum based implementation as single target, the output should be same except that the
        # output should be a list of explanations in the case of multi-target
        kwargs = {**base_setup_kwargs}
        kwargs["explainer_class"] = explainer_class
        single_target_explanation_ouputs = []
        for target in tqdm.tqdm(targets):
            kwargs["target"] = target
            set_all_random_seeds(1234)
            single_target_explanations = self.basic_model_assert(
                **kwargs,
                delta=delta,
                internal_batch_size=internal_batch_size,
            )
            if isinstance(target, list):
                kwargs["target"] = (
                    [target] if isinstance(target[0], (int, tuple)) else target
                )
            elif isinstance(target, int):
                kwargs["target"] = [target]
            set_all_random_seeds(1234)
            multi_target_explanations = self.basic_model_assert(
                **kwargs,
                is_multi_target=True,
                delta=delta,
                internal_batch_size=internal_batch_size,
            )
            self.compare_explanation_per_target(
                multi_target_explanations[0],
                single_target_explanations,
            )
            single_target_explanation_ouputs.append(single_target_explanations)

        # now test the case where the targets are sent as a list to multi-target and the same targets and all
        # the explanations generated by our multi-target implementation should match the explanations generated
        # by the captum-based implementation
        kwargs["target"] = targets
        set_all_random_seeds(1234)
        multi_target_explanations = self.basic_model_assert(
            **kwargs,
            is_multi_target=True,
            delta=delta,
            internal_batch_size=internal_batch_size,
        )
        for multi_target_explanation, single_target_explanation in zip(
            multi_target_explanations, single_target_explanation_ouputs
        ):
            # target explanation in the list should match the single target explanations at the same index
            self.compare_explanation_per_target(
                multi_target_explanation,
                single_target_explanation,
            )

    def basic_model_assert(
        self,
        explainer_class: Type[Explainer],
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Optional[TensorOrTupleOfTensorsGeneric] = None,
        feature_mask: Optional[Tensor] = None,
        train_baselines: Optional[TensorOrTupleOfTensorsGeneric] = None,
        expected_explanation: Optional[Tensor] = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        is_multi_target: bool = False,
        assert_failure: bool = False,
        delta: float = 1e-5,
        internal_batch_size: Optional[int] = None,
    ) -> Tensor:
        explainer = explainer_class(
            model,
            is_multi_target=is_multi_target,
            internal_batch_size=internal_batch_size,
        )
        if assert_failure:
            self.assertRaises(
                AssertionError,
                explainer.explain,
                inputs,
                target=target,
                additional_forward_args=additional_forward_args,
            )
            return
        else:
            kwargs = {"baselines": baselines} if baselines is not None else {}
            kwargs = (
                {"train_baselines": train_baselines}
                if train_baselines is not None
                else kwargs
            )
            kwargs = (
                {"feature_mask": feature_mask} if feature_mask is not None else kwargs
            )
            explanation = explainer.explain(
                inputs,
                target=target,
                additional_forward_args=additional_forward_args,
                **kwargs,
            )

        if expected_explanation is not None:
            if is_multi_target:
                assert isinstance(explanation, list)
                assert len(explanation) == len(expected_explanation)

                for output_per_target, expected_per_target in zip(
                    explanation, expected_explanation
                ):
                    self.compare_explanation_per_target(
                        output_per_target, expected_per_target, delta=delta
                    )
            else:
                self.compare_explanation_per_target(
                    explanation, expected_explanation, delta=delta
                )
        return explanation

    def compare_explanation_per_target(
        self,
        output_explanation_per_target: Tuple[Tensor, ...],
        expected_explanation_per_target: Tuple[Tensor, ...],
        delta: float = 1e-5,
    ) -> None:
        if not isinstance(output_explanation_per_target, tuple):
            output_explanation_per_target = (output_explanation_per_target,)
        if not isinstance(expected_explanation_per_target, tuple):
            expected_explanation_per_target = (expected_explanation_per_target,)

        for output_explanation_per_input, expected_explanation_per_input in zip(
            output_explanation_per_target, expected_explanation_per_target
        ):
            print(
                output_explanation_per_input,
                expected_explanation_per_input,
                output_explanation_per_input - expected_explanation_per_input,
            )
            assertTensorAlmostEqual(
                self,
                output_explanation_per_input,
                expected_explanation_per_input,
                delta=delta,
            )
