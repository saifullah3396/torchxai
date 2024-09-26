import logging
import unittest
from logging import getLogger
from pathlib import Path
from typing import List, Union

import torch
import tqdm
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from torch import Tensor, nn

from tests.helpers.basic import assertTensorAlmostEqual, set_all_random_seeds
from tests.helpers.basic_models import MNISTCNNModel, MNISTLinearModel
from tests.metrics.base import MetricTestsBase
from torchxai.explanation_framework.explainers.factory import ExplainerFactory
from torchxai.explanation_framework.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.explanation_framework.utils.common import grid_segmenter
from torchxai.metrics import input_invariance
from torchxai.metrics._utils.visualization import visualize_attribution

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class Test(MetricTestsBase):
    def mnist_setup(
        self,
        model_type: str = "linear",
        explainer: str = "saliency",
        explainer_kwargs: dict = {},
        train_and_eval_model: bool = True,
    ) -> None:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.datasets import MNIST

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataset = MNIST(
            root="/tmp/mnist/",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
            target_transform=transforms.Compose([lambda x: torch.tensor(x)]),
        )
        test_dataset = MNIST(
            root="/tmp/mnist/",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
            target_transform=transforms.Compose([lambda x: torch.tensor(x)]),
        )
        train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        train_baselines = next(iter(train_dataloader))[0].to(device)

        if model_type == "linear":
            model = MNISTLinearModel()
            input_layer_names = ["fc1"]
        elif model_type == "cnn":
            model = MNISTCNNModel()
            input_layer_names = ["conv1"]
        else:
            raise ValueError("Invalid model")

        def train_model(model, dataloader):
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            model.train()
            for epoch in tqdm.tqdm(range(10)):
                for i, (images, labels) in enumerate(dataloader):
                    optimizer.zero_grad()
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                if epoch % 1 == 0:
                    print(f"Epoch {epoch}, Step {i}, Loss {loss.item()}")

        # evaluate
        def evaluate_model(model, dataloader, input_shift: float = 0):
            device = "cuda:0"
            model.eval()
            model.to(device)
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(device) + input_shift
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            print(
                f"MNIST {model_type} model trained with accuracy: {100 * correct / total}"
            )

        model.to(device)
        model.eval()

        if train_and_eval_model:
            model_path = f"/tmp/mnist/{model_type}_model.pth"
            if Path(model_path).exists():
                model.load_state_dict(torch.load(model_path))
            else:
                train_model(model, train_dataloader)
                torch.save(model.state_dict(), model_path)
            evaluate_model(model, test_dataloader)

        batch = next(iter(test_dataloader))
        inputs = batch[0].to(device)
        target = batch[1].to(device)

        explanation_func = ExplainerFactory.create(explainer, model, **explainer_kwargs)
        kwargs = {
            "inputs": inputs,
            "target": target,
            "explainer": explanation_func,
            "train_baselines": train_baselines,
            "input_layer_names": input_layer_names,
        }
        return kwargs

    def test_input_invariance_mnist_linear_saliency(self) -> None:
        # this setup is exactly the same as in the paper: https://arxiv.org/pdf/1711.00867
        # a 3-layer linear model is trained on MNIST, input invariance is computed for saliency maps
        # on 4 input samples. The expected output is [True, True, True, True]
        kwargs = self.mnist_setup(
            model_type="linear",
            explainer="saliency",
            train_and_eval_model=True,
        )
        kwargs["constant_shifts"] = torch.ones(
            1, 28, 28, device=kwargs["inputs"].device
        )
        kwargs.pop("train_baselines")

        # test both with FusionExplainer and Captum Attribution
        # for FusionExplainer, we need to pass the explainer object
        self.output_assert(expected=torch.tensor([True] * 4), **kwargs)

        # for Captum Attribution, we need to pass the underling captumn explainer object
        kwargs["explainer"] = kwargs["explainer"]._explanation_fn
        self.output_assert(expected=torch.tensor([True] * 4), **kwargs)

    def test_input_invariance_mnist_linear_input_x_gradient(self) -> None:
        # this setup is exactly the same as in the paper: https://arxiv.org/pdf/1711.00867
        # a 3-layer linear model is trained on MNIST, input invariance is computed for input_x_gradient
        # on 4 input samples. The expected output is [False, False, False, False]
        kwargs = self.mnist_setup(
            model_type="linear",
            explainer="input_x_gradient",
            train_and_eval_model=True,
        )
        kwargs["constant_shifts"] = torch.ones(
            1, 28, 28, device=kwargs["inputs"].device
        )
        kwargs.pop("train_baselines")

        # test both with FusionExplainer and Captum Attribution
        # for FusionExplainer, we need to pass the explainer object
        self.output_assert(expected=torch.tensor([False] * 4), **kwargs)

        # for Captum Attribution, we need to pass the underling captumn explainer object
        kwargs["explainer"] = kwargs["explainer"]._explanation_fn
        self.output_assert(expected=torch.tensor([False] * 4), **kwargs)

    def test_input_invariance_mnist_linear_integrated_gradients_baselines_zero(
        self,
    ) -> None:
        # this setup is exactly the same as in the paper: https://arxiv.org/pdf/1711.00867
        # a 3-layer linear model is trained on MNIST, input invariance is computed for integrated_gradients
        # on 4 input samples. The expected output is [False, False, False, False] with zero_baseline
        kwargs = self.mnist_setup(
            model_type="linear",
            explainer="integrated_gradients",
            train_and_eval_model=True,
            explainer_kwargs={"n_steps": 200},
        )
        kwargs["constant_shifts"] = torch.ones(
            1, 28, 28, device=kwargs["inputs"].device
        )
        kwargs["baselines"] = 0
        kwargs["shifted_baselines"] = 0
        kwargs.pop("train_baselines")

        # test both with FusionExplainer and Captum Attribution
        # for FusionExplainer, we need to pass the explainer object
        self.output_assert(expected=torch.tensor([False] * 4), **kwargs)

        # for Captum Attribution, we need to pass the underling captumn explainer object
        kwargs["explainer"] = kwargs["explainer"]._explanation_fn
        self.output_assert(expected=torch.tensor([False] * 4), **kwargs)

    def test_input_invariance_mnist_linear_integrated_gradients_baselines_black(
        self,
    ) -> None:
        # this setup is exactly the same as in the paper: https://arxiv.org/pdf/1711.00867
        # a 3-layer linear model is trained on MNIST, input invariance is computed for integrated_gradients
        # on 4 input samples. The expected output is [True, True, True, True] with black_baseline
        kwargs = self.mnist_setup(
            model_type="linear",
            explainer="integrated_gradients",
            train_and_eval_model=True,
            explainer_kwargs={"n_steps": 200},
        )
        kwargs["constant_shifts"] = torch.ones(
            1, 28, 28, device=kwargs["inputs"].device
        )
        kwargs.pop("train_baselines")
        kwargs["baselines"] = 0
        kwargs["shifted_baselines"] = -1

        # test both with FusionExplainer and Captum Attribution
        # for FusionExplainer, we need to pass the explainer object
        self.output_assert(expected=torch.tensor([True] * 4), **kwargs)

        # for Captum Attribution, we need to pass the underling captumn explainer object
        kwargs["explainer"] = kwargs["explainer"]._explanation_fn
        self.output_assert(expected=torch.tensor([True] * 4), **kwargs)

    def test_input_invariance_mnist_linear_occlusion_baselines_black(
        self,
    ) -> None:
        # here apply the same logic as in the paper: https://arxiv.org/pdf/1711.00867
        # a 3-layer linear model is trained on MNIST, input invariance is computed for occlusion
        # on 4 input samples. The expected output is [True, True, True, True] with black_baseline and
        # atol=1e-3. Note that these results were not in the paper, so this shows how the implementation
        # can be used for other explainers
        explainer_kwargs = dict(sliding_window_shapes=(1, 4, 4))
        kwargs = self.mnist_setup(
            model_type="linear",
            explainer="occlusion",
            train_and_eval_model=True,
            explainer_kwargs=explainer_kwargs,
        )
        kwargs["constant_shifts"] = torch.ones(
            1, 28, 28, device=kwargs["inputs"].device
        )
        kwargs.pop("train_baselines")
        kwargs["baselines"] = 0
        kwargs["shifted_baselines"] = -1
        kwargs["atol"] = 1e-3

        # test both with FusionExplainer and Captum Attribution
        # for FusionExplainer, we need to pass the explainer object
        self.output_assert(expected=torch.tensor([True] * 4), **kwargs)

        # for Captum Attribution, we need to pass the underling captumn explainer object
        kwargs["explainer"] = kwargs["explainer"]._explanation_fn
        self.output_assert(
            expected=torch.tensor([True] * 4), **kwargs, **explainer_kwargs
        )

    def test_input_invariance_mnist_linear_lime_baselines_black(
        self,
    ) -> None:
        # here apply the same logic as in the paper: https://arxiv.org/pdf/1711.00867
        # a 3-layer linear model is trained on MNIST, input invariance is computed for lime
        # on 4 input samples. The expected output is [True, True, True, True] with black_baseline and
        # atol=1e-1. Note that these results were not in the paper, so this shows how the implementation
        # can be used for other explainers
        explainer_kwargs = {"n_samples": 200}
        kwargs = self.mnist_setup(
            model_type="linear",
            explainer="lime",
            train_and_eval_model=True,
            explainer_kwargs=explainer_kwargs,
        )
        kwargs["constant_shifts"] = torch.ones(
            1, 28, 28, device=kwargs["inputs"].device
        )
        kwargs.pop("train_baselines")
        kwargs["baselines"] = 0
        kwargs["shifted_baselines"] = -1
        kwargs["atol"] = 1e-1
        kwargs["feature_mask"] = grid_segmenter(kwargs["inputs"], 4)

        # test both with FusionExplainer and Captum Attribution
        # for FusionExplainer, we need to pass the explainer object
        # in FusionExplainer, we also weight the attributions on output but here we turn it off to keep the
        # results same as captum implementation
        # note that while this fails the test on some cases, visually the results are very similar which can be
        # seen by setting visualize=True
        self.output_assert(
            expected=torch.tensor([True, True, True, False]),
            weight_attributions=False,
            **kwargs,
        )

        # for Captum Attribution, we need to pass the underling captumn explainer object
        # note that while this fails the test on some cases, visually the results are very similar which can be
        # seen by setting visualize=True
        kwargs["explainer"] = kwargs["explainer"]._explanation_fn
        self.output_assert(
            expected=torch.tensor([True, True, True, False]),
            **kwargs,
            **explainer_kwargs,
        )

    def output_assert(
        self,
        expected: Tensor,
        explainer: Union[Attribution, FusionExplainer],
        inputs: TensorOrTupleOfTensorsGeneric,
        constant_shifts: TensorOrTupleOfTensorsGeneric,
        input_layer_names: List[str],
        visualize: bool = False,
        **kwargs,
    ) -> Tensor:
        set_all_random_seeds(1234)
        invariance, expl_inputs, expl_shifted_inputs = input_invariance(
            explainer=explainer,
            inputs=inputs,
            constant_shifts=constant_shifts,
            input_layer_names=input_layer_names,
            **kwargs,
        )
        assertTensorAlmostEqual(self, invariance.float(), expected.float())

        if visualize:
            # here explanations can be visualized for debugging purposes
            for input, expl_input, expl_shifted_input in zip(
                inputs, expl_inputs, expl_shifted_inputs
            ):
                visualize_attribution(input, expl_input, "Original")
                visualize_attribution(input, expl_shifted_input, "Shifted")

        return invariance


if __name__ == "__main__":
    unittest.main()
