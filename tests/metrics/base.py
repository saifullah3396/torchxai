import logging
from logging import getLogger
from typing import List, Tuple

import torch
from captum.attr import DeepLift, IntegratedGradients

from tests.helpers import BaseTest
from tests.helpers.basic_models import (
    BasicModel2,
    BasicModel4_MultiArgs,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
)

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class MetricTestsBase(BaseTest):
    def basic_single_setup(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([3.0])
        input2 = torch.tensor([1.0])
        inputs = (input1, input2)
        return dict(
            model=model, inputs=inputs, attribution_fn=IntegratedGradients(model)
        )

    def basic_batch_setup(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([3.0] * 3)
        input2 = torch.tensor([1.0] * 3)
        inputs = (input1, input2)
        return dict(
            model=model, inputs=inputs, attribution_fn=IntegratedGradients(model)
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
            attribution_fn=IntegratedGradients(model),
        )

    def classification_convnet_multi_targets_setup(self):
        model = BasicModel_ConvNet_One_Conv()
        inputs = torch.stack([torch.arange(1, 17).float()] * 20, dim=0).view(
            20, 1, 4, 4
        )
        target = torch.tensor([1] * 20)
        return dict(
            model=model, inputs=inputs, attribution_fn=DeepLift(model), target=target
        )

    def classification_tpl_target_setup(self):
        model = BasicModel_MultiLayer()
        inputs = torch.arange(1.0, 13.0).view(4, 3)
        additional_forward_args = (torch.arange(1, 13).view(4, 3).float(), True)
        target: List[Tuple[int, ...]] = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
        return dict(
            model=model,
            inputs=inputs,
            attribution_fn=IntegratedGradients(model),
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
            attribution_fn=IntegratedGradients(model),
            additional_forward_args=additional_forward_args,
            target=targets,
            baselines=baselines,
        )
