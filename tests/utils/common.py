import logging
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import tqdm
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from ignite.utils import convert_tensor
from torch import Tensor, nn

from tests.helpers.basic_models import MNISTCNNModel, MNISTLinearModel
from tests.utils.containers import TestBaseConfig
from torchxai.explainers.explainer import Explainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def grid_segmenter(images: torch.Tensor, cell_size: int = 16) -> torch.Tensor:
    feature_mask = []
    for image in images:
        # image dimensions are C x H x H
        dim_x, dim_y = image.shape[1] // cell_size, image.shape[2] // cell_size
        mask = (
            torch.arange(dim_x * dim_y, device=images.device)
            .view((dim_x, dim_y))
            .repeat_interleave(cell_size, dim=0)
            .repeat_interleave(cell_size, dim=1)
            .long()
            .unsqueeze(0)
        )
        feature_mask.append(mask)
    return torch.stack(feature_mask)


def train_mnist_model(model, dataloader, n_epochs: int = 10, lr: float = 0.01):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    model.train()
    pbar = tqdm.tqdm(range(n_epochs))
    for epoch in pbar:
        for i, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            pbar.set_postfix_str(f"Epoch {epoch}, Loss {loss.item()}")


def evaluate_mnist_model(model, dataloader):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logger.info(f"Model accuracy: {100 * correct / total}")


def mnist_dataloader():
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
    return train_dataloader, test_dataloader, train_baselines


def mnist_trainer(model_type: bool = "linear", train_and_eval_model: bool = True):
    train_dataloader, test_dataloader, train_baselines = mnist_dataloader()
    if model_type == "linear":
        model = MNISTLinearModel()
        input_layer_names = ["fc1"]
    elif model_type == "cnn":
        model = MNISTCNNModel()
        input_layer_names = ["conv1"]
    else:
        raise ValueError("Invalid model")

    # load model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if train_and_eval_model:
        model_path = f"/tmp/mnist/{model_type}_model.pth"
        if Path(model_path).exists():
            logger.info(f"Loading {model_type} model for tests from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            logger.info(
                f"Training {model_type} model for tests and caching to {model_path}"
            )
            train_mnist_model(model, train_dataloader)
            torch.save(model.state_dict(), model_path)
        evaluate_mnist_model(model, test_dataloader)

    model.eval()
    batch = next(iter(test_dataloader))
    inputs = batch[0].to(device)
    target = batch[1].to(device)
    train_baselines = train_baselines.to(device)

    return TestBaseConfig(
        model=model,
        inputs=inputs,
        additional_forward_args=None,
        target=target,
        train_baselines=train_baselines,
        n_features=(1 * 28 * 28),
        input_layer_names=input_layer_names,
    )


def compare_explanation_per_target(
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
        assert_tensor_almost_equal(
            output_explanation_per_input,
            expected_explanation_per_input,
            delta=delta,
            mode="mean",
        )


def compute_explanations(
    explainer: Union[Explainer, Attribution],
    inputs: TensorOrTupleOfTensorsGeneric,
    additional_forward_args: Optional[Any] = None,
    baselines: Optional[BaselineType] = None,
    train_baselines: Optional[BaselineType] = None,
    feature_mask: Optional[TensorOrTupleOfTensorsGeneric] = None,
    target: Optional[TargetType] = None,
    multiply_by_inputs: bool = False,
    use_captum_explainer: bool = False,
    device="cpu",
    **explainer_kwargs,
) -> Tensor:
    if use_captum_explainer:
        explainer = explainer._explanation_fn

    inputs = convert_tensor(inputs, device=device)
    if target is not None:
        target = (
            convert_tensor(target, device=device)
            if isinstance(target, torch.Tensor)
            or (isinstance(target, list) and isinstance(target[0], torch.Tensor))
            else target
        )
    if additional_forward_args is not None:
        if isinstance(additional_forward_args, tuple):
            additional_forward_args = tuple(
                arg.to(device) if isinstance(arg, torch.Tensor) else arg
                for arg in additional_forward_args
            )
        else:
            additional_forward_args = (
                additional_forward_args.to(device)
                if isinstance(additional_forward_args, torch.Tensor)
                else additional_forward_args
            )
    if baselines is not None:
        baselines = convert_tensor(baselines, device=device)
    if feature_mask is not None:
        feature_mask = convert_tensor(feature_mask, device=device)
    if train_baselines is not None:
        train_baselines = convert_tensor(train_baselines, device=device)

    if isinstance(explainer, Explainer):
        # our own explainer does not take explainer configuration specific kwargs at inference
        # but rather takes it in initialization
        explainer_kwargs = {}
        if baselines is not None:
            explainer_kwargs["baselines"] = baselines
        if feature_mask is not None:
            explainer_kwargs["feature_mask"] = feature_mask
        if train_baselines is not None:
            explainer_kwargs["train_baselines"] = train_baselines
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
            **explainer_kwargs,
        )
    else:
        raise AssertionError(
            "Explainer must be an instance of Explainer or Attribution."
        )
    if multiply_by_inputs:
        explanations = cast(
            TensorOrTupleOfTensorsGeneric,
            tuple(attr / input for input, attr in zip(inputs, explanations)),
        )
    return explanations


def assert_tensor_almost_equal(
    actual,
    expected,
    delta: float = 0.0001,
    mode: str = "sum",
) -> None:
    assert isinstance(actual, torch.Tensor), (
        "Actual parameter given for " "comparison must be a tensor."
    )
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)
    assert (
        actual.shape == expected.shape
    ), f"Expected tensor with shape: {expected.shape}. Actual shape {actual.shape}."
    actual = actual.cpu()
    expected = expected.cpu()

    # check if both are nan
    if torch.isnan(actual).all():
        assert torch.isnan(
            expected
        ).all(), f"Actual tensor is nan while expected tensor is not. Actual: {actual}, Expected: {expected}"
        return

    if mode == "sum":
        assert (
            torch.sum(torch.abs(actual - expected)).item() < delta
        ), f"Tensors are not equal with tolerance ({delta}). Actual: {actual}, Expected: {expected}"
    elif mode == "mean":
        assert (
            torch.mean(torch.abs(actual - expected)).item() < delta
        ), f"Tensors are not equal with tolerance ({delta}). Actual: {actual}, Expected: {expected}"
    elif mode == "max":
        # if both tensors are empty, they are equal but there is no max
        if actual.numel() == expected.numel() == 0:
            return

        if actual.size() == torch.Size([]):
            assert (
                torch.max(torch.abs(actual - expected)).item() < delta
            ), f"Tensors are not equal with tolerance ({delta}). Actual: {actual}, Expected: {expected}"
        else:
            for index, (input, ref) in enumerate(zip(actual, expected)):
                almost_equal = abs(input - ref) <= delta
                if hasattr(almost_equal, "__iter__"):
                    almost_equal = almost_equal.all()
                assert (
                    almost_equal
                ), "Values at index {}, {} and {}, differ more than by {}".format(
                    index, input, ref, delta
                )
    else:
        raise ValueError("Mode for assertion comparison must be one of `max` or `sum`.")


def assert_all_tensors_almost_equal(tensors: List[torch.Tensor]):
    for i in range(1, len(tensors)):
        if isinstance(tensors[i], list):
            for x, y in zip(tensors[i - 1], tensors[i]):
                assert_tensor_almost_equal(x, y)
        else:
            assert_tensor_almost_equal(tensors[i - 1], tensors[i])


def set_all_random_seeds(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
