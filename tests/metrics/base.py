import logging
from typing import Any, List, Optional, Tuple, Union, cast

import torch
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from torch import Tensor

from tests.helpers import BaseTest
from tests.helpers.basic_models import (
    BasicModel2,
    BasicModel4_MultiArgs,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    ParkFunction,
)
from torchxai.explanation_framework.explainers._grad.deeplift import DeepLiftExplainer
from torchxai.explanation_framework.explainers._grad.integrated_gradients import (
    IntegratedGradientsExplainer,
)
from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricTestsBase(BaseTest):
    def park_function_setup(self) -> None:
        model = ParkFunction()
        input = torch.tensor([[0.24, 0.48, 0.56, 0.99, 0.68, 0.86]])
        return dict(
            model=model, inputs=input, explainer=IntegratedGradientsExplainer(model)
        )

    def basic_single_setup(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([3.0])
        input2 = torch.tensor([1.0])
        inputs = (input1, input2)
        return dict(
            model=model, inputs=inputs, explainer=IntegratedGradientsExplainer(model)
        )

    def basic_batch_setup(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([3.0] * 3)
        input2 = torch.tensor([1.0] * 3)
        inputs = (input1, input2)
        return dict(
            model=model, inputs=inputs, explainer=IntegratedGradientsExplainer(model)
        )

    def basic_additional_forward_args_setup(self):
        model = BasicModel4_MultiArgs()
        input1 = torch.tensor([[1.5, 2.0, 3.3]])
        input2 = torch.tensor([[3.0, 3.5, 2.2]])
        inputs = (input1, input2)
        args = torch.tensor([[1.0, 3.0, 4.0]])
        return dict(
            model=model,
            inputs=inputs,
            additional_forward_args=(args,),
            explainer=IntegratedGradientsExplainer(model),
        )

    def classification_convnet_multi_targets_setup(self):
        model = BasicModel_ConvNet_One_Conv()
        inputs = torch.stack([torch.arange(1, 17).float()] * 20, dim=0).view(
            20, 1, 4, 4
        )
        target = torch.tensor([1] * 20)
        return dict(
            model=model,
            inputs=inputs,
            explainer=DeepLiftExplainer(model),
            target=target,
        )

    def classification_tpl_target_setup(self):
        model = BasicModel_MultiLayer()
        inputs = torch.arange(1.0, 13.0).view(4, 3)
        additional_forward_args = (torch.arange(1, 13).view(4, 3).float(), True)
        target: List[Tuple[int, ...]] = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
        return dict(
            model=model,
            inputs=inputs,
            explainer=IntegratedGradientsExplainer(model),
            additional_forward_args=additional_forward_args,
            target=target,
        )

    def classification_tpl_target_w_baseline_setup(self):
        model = BasicModel_MultiLayer()
        inputs = torch.arange(1.0, 13.0).view(4, 3)
        baselines = torch.ones(4, 3)
        additional_forward_args = (torch.arange(1, 13).view(4, 3).float(), True)
        targets: List[Tuple[int, ...]] = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
        return dict(
            model=model,
            inputs=inputs,
            explainer=IntegratedGradientsExplainer(model),
            additional_forward_args=additional_forward_args,
            target=targets,
            baselines=baselines,
        )

    def compute_explanations(
        self,
        explainer: Union[FusionExplainer, Attribution],
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: Optional[Any] = None,
        baselines: Optional[BaselineType] = None,
        target: Optional[TargetType] = None,
        multiply_by_inputs: bool = False,
    ) -> Tensor:
        if isinstance(explainer, FusionExplainer):
            explainer_kwargs = {}
            if baselines is not None:
                explainer_kwargs["baselines"] = baselines

            explanations = explainer.explain(
                inputs,
                additional_forward_args=additional_forward_args,
                target=target,
                **explainer_kwargs,
            )
        elif isinstance(explainer, Attribution):
            explanations = explainer.attribute(
                inputs,
                baselines=baselines,
                additional_forward_args=additional_forward_args,
                target=target,
            )
        if multiply_by_inputs:
            explanations = cast(
                TensorOrTupleOfTensorsGeneric,
                tuple(attr / input for input, attr in zip(inputs, explanations)),
            )
        return explanations
