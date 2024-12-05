import logging

import pytest
import torch

from tests.helpers.basic_models import (
    BasicModel2,
    BasicModel4_MultiArgs,
    BasicModel7_ReluMultiTensor,
    BasicModel7_SumMultiTensor,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    ParkFunction,
)
from tests.helpers.classification_models import (
    SigmoidModel,
    SoftmaxModel,
    SoftmaxModelTupleInput,
)
from tests.utils.common import compute_explanations, mnist_trainer, set_all_random_seeds
from tests.utils.containers import TestBaseConfig, TestRuntimeConfig
from torchxai.explainers.factory import ExplainerFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pytest_runtest_setup():
    set_all_random_seeds(1234)


@pytest.fixture()
def park_function_configuration():
    yield TestBaseConfig(
        model=ParkFunction(),
        inputs=torch.tensor([[0.24, 0.48, 0.56, 0.99, 0.68, 0.86]]),
        n_features=6,
    )


@pytest.fixture()
def basic_model_single_input_config():
    yield TestBaseConfig(
        model=BasicModel2(),
        inputs=(
            torch.tensor([3.0]),
            torch.tensor([1.0]),
        ),
        n_features=2,
    )


@pytest.fixture()
def basic_model_single_batched_input_config():
    yield TestBaseConfig(
        model=BasicModel2(),
        inputs=(
            torch.tensor([[3.0]]),
            torch.tensor([[1.0]]),
        ),
        n_features=2,
    )


@pytest.fixture()
def basic_model_batch_input_config():
    yield TestBaseConfig(
        model=BasicModel2(),
        inputs=(
            torch.tensor([3.0] * 3),
            torch.tensor([1.0] * 3),
        ),
        n_features=2,
    )


@pytest.fixture()
def basic_model_batch_input_with_additional_forward_args_config():
    yield TestBaseConfig(
        model=BasicModel4_MultiArgs(),
        inputs=(
            torch.tensor([[1.5, 2.0, 3.3]]),
            torch.tensor([[3.0, 3.5, 2.2]]),
        ),
        additional_forward_args=torch.tensor([[1.0, 3.0, 4.0]]),
        n_features=6,
    )


@pytest.fixture()
def classification_convnet_model_with_multiple_targets_config():
    yield TestBaseConfig(
        model=BasicModel_ConvNet_One_Conv(),
        inputs=torch.stack([torch.arange(1, 17).float()] * 20, dim=0).view(20, 1, 4, 4),
        target=torch.tensor([1] * 20),
        n_features=(1 * 4 * 4),
    )


@pytest.fixture()
def classification_multilayer_model_with_tuple_targets_config():
    yield TestBaseConfig(
        model=BasicModel_MultiLayer(),
        inputs=torch.arange(1.0, 13.0).view(4, 3).float(),
        additional_forward_args=(torch.arange(1, 13).view(4, 3).float(), True),
        target=[(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
        n_features=(3),
    )


@pytest.fixture()
def classification_multilayer_model_with_baseline_and_tuple_targets_config():
    yield TestBaseConfig(
        model=BasicModel_MultiLayer(),
        inputs=torch.arange(1.0, 13.0).view(4, 3).float(),
        additional_forward_args=(torch.arange(1, 13).view(4, 3).float(), True),
        target=[(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)],
        baselines=torch.ones(4, 3),
        n_features=(3),
    )


@pytest.fixture()
def classification_sigmoid_model_single_input_single_target_config():
    yield TestBaseConfig(
        model=SigmoidModel(10, 20, 10),
        inputs=torch.tensor([[1.0] * 10]),
        target=torch.tensor([1]),
    )


@pytest.fixture()
def classification_softmax_model_single_input_single_target_config():
    yield TestBaseConfig(
        model=SoftmaxModel(10, 20, 10),
        inputs=torch.tensor([[1.0] * 10]),
        target=torch.tensor([1]),
    )


@pytest.fixture()
def classification_softmax_model_multi_input_single_target_config():
    yield TestBaseConfig(
        model=SoftmaxModel(10, 20, 10),
        inputs=torch.tensor([[1.0] * 10] * 3),
        target=torch.tensor([1]),
    )


@pytest.fixture()
def classification_softmax_model_multi_tuple_input_single_target_config():
    yield TestBaseConfig(
        model=SoftmaxModelTupleInput(10, 20, 10),
        inputs=(torch.tensor([[1.0] * 10] * 3), torch.tensor([[-1.0] * 10] * 3)),
        target=torch.tensor([1]),
    )


@pytest.fixture()
def classification_alexnet_model_single_sample_config():
    from torchvision.models import alexnet

    model = alexnet(pretrained=True)
    model.eval()
    model.zero_grad()
    yield TestBaseConfig(
        model=model, inputs=torch.randn(1, 3, 224, 224), target=torch.tensor([1])
    )


@pytest.fixture()
def classification_alexnet_model_config():
    from torchvision.models import alexnet

    model = alexnet(pretrained=True)
    model.eval()
    model.zero_grad()
    yield TestBaseConfig(
        model=model, inputs=torch.randn(10, 3, 224, 224), target=torch.tensor([1])
    )


@pytest.fixture()
def classification_alexnet_model_real_images_single_sample_config():
    from io import BytesIO

    import requests
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from torchvision.models import alexnet

    image_urls = [
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01440764_tench.JPEG?raw=true",
    ]
    labels = [0]

    images = []
    for url in image_urls:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize the image if needed
                transforms.ToTensor(),  # Convert to a tensor (normalizes pixel values to [0, 1])
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize pixel values to ImageNet values
            ]
        )
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(
            0
        )  # Shape: [1, 3, 256, 256] for a batch of 1 image
        if images == []:
            images = image_tensor
        else:
            images = torch.cat((images, image_tensor), dim=0)
    labels = torch.tensor(labels)
    model = alexnet(pretrained=True)
    model.eval()
    model.zero_grad()
    yield TestBaseConfig(model=model, inputs=images, target=labels)


@pytest.fixture()
def classification_alexnet_model_real_images_config():
    from io import BytesIO

    import requests
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from torchvision.models import alexnet

    image_urls = [
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01440764_tench.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01537544_indigo_bunting.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01641577_bullfrog.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01693334_green_lizard.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01819313_sulphur-crested_cockatoo.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01883070_wombat.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n01990800_isopod.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02091467_Norwegian_elkhound.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02099429_curly-coated_retriever.JPEG?raw=true",
        "https://github.com/EliSchwartz/imagenet-sample-images/blob/master/n02113624_toy_poodle.JPEG?raw=true",
    ]
    labels = [0, 14, 30, 46, 89, 106, 126, 174, 206, 265]

    images = []
    for url in image_urls:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize the image if needed
                transforms.ToTensor(),  # Convert to a tensor (normalizes pixel values to [0, 1])
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize pixel values to ImageNet values
            ]
        )
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(
            0
        )  # Shape: [1, 3, 256, 256] for a batch of 1 image
        if images == []:
            images = image_tensor
        else:
            images = torch.cat((images, image_tensor), dim=0)
    labels = torch.tensor(labels)
    model = alexnet(pretrained=True)
    model.eval()
    model.zero_grad()
    yield TestBaseConfig(model=model, inputs=images, target=labels)


@pytest.fixture()
def mnist_train_configuration():
    def _mnist_train_configuration(model_type: str, train_and_eval_model: bool):
        return mnist_trainer(model_type, train_and_eval_model)

    yield _mnist_train_configuration


@pytest.fixture()
def metrics_runtime_test_configuration(request):
    runtime_config: TestRuntimeConfig = request.param
    base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)
    if runtime_config.override_target is not None:
        base_config.target = runtime_config.override_target
    base_config.model.eval()
    base_config.put_to_device(runtime_config.device)
    explainer = ExplainerFactory.create(
        runtime_config.explainer, base_config.model, **runtime_config.explainer_kwargs
    )
    explanations = compute_explanations(
        explainer=explainer,
        inputs=base_config.inputs,
        additional_forward_args=base_config.additional_forward_args,
        baselines=base_config.baselines,
        train_baselines=base_config.train_baselines,
        feature_mask=base_config.feature_mask,
        target=base_config.target,
        multiply_by_inputs=base_config.multiply_by_inputs,
        use_captum_explainer=runtime_config.use_captum_explainer,
        **runtime_config.explainer_kwargs,
    )
    yield base_config, runtime_config, explanations


@pytest.fixture()
def explainer_runtime_test_configuration(request):
    runtime_config: TestRuntimeConfig = request.param
    base_config: TestBaseConfig = request.getfixturevalue(runtime_config.target_fixture)
    if runtime_config.override_target is not None:
        base_config.target = runtime_config.override_target

    yield base_config, runtime_config


@pytest.fixture()
def multi_modal_sequence_sum():
    def test_sequence_tensor(size=12, embedding_size=3):
        return (
            torch.tensor([0] + list(range(3, size + 3)) + [1, 2])
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, embedding_size, size + 3)
            .repeat(1, 1, 1)
            .permute(0, 2, 1)
        ).float()

    def test_image(size=9):
        return (
            torch.arange(size)
            .view(1, 1, 3, 3)
            .repeat_interleave(2, dim=-1)
            .repeat_interleave(2, dim=-2)
            .float()
        )

    size = 6
    sequence1 = test_sequence_tensor(size)
    sequence2 = test_sequence_tensor(size) + size + 3
    sequence3 = test_sequence_tensor(size) + (size + 3) * 2
    image1 = test_image() + (size + 3) * 3
    feature_mask = (
        sequence1.clone().long(),
        sequence2.clone().long(),
        sequence3.clone().long(),
        image1.clone().long(),
    )
    frozen_features = torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])
    n_features = (
        torch.cat([x.flatten() for x in feature_mask]).unique().numel()
        - frozen_features.numel()
    )
    inputs = (sequence1, sequence2, sequence3, image1)
    total_sum = sum(x.sum() for x in inputs)
    inputs = tuple(x / total_sum for x in inputs)
    target = None

    yield TestBaseConfig(
        model=BasicModel7_SumMultiTensor(),
        inputs=inputs,
        target=target,
        feature_mask=feature_mask,
        baselines=tuple(torch.zeros_like(x) for x in inputs),
        frozen_features=[torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])],
        n_features=n_features,  # total features is actually
    )


@pytest.fixture()
def multi_modal_sequence_relu():
    def test_sequence_tensor(size=12, embedding_size=4):
        return (
            torch.tensor([0] + list(range(3, size + 3)) + [1, 2])
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, embedding_size, size + 3)
            .repeat(1, 1, 1)
            .permute(0, 2, 1)
        ).float()

    def test_image(size=9):
        return (
            torch.arange(size)
            .view(1, 1, 3, 3)
            .repeat_interleave(2, dim=-1)
            .repeat_interleave(2, dim=-2)
            .float()
        )

    size = 6
    sequence1 = test_sequence_tensor(size)
    sequence2 = test_sequence_tensor(size) + size + 3
    sequence3 = test_sequence_tensor(size) + (size + 3) * 2
    image1 = test_image() + (size + 3) * 3
    feature_mask = (
        sequence1.clone().long(),
        sequence2.clone().long(),
        sequence3.clone().long(),
        image1.clone().long(),
    )
    frozen_features = torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])
    n_features = (
        torch.cat([x.flatten() for x in feature_mask]).unique().numel()
        - frozen_features.numel()
    )
    inputs = (sequence1, sequence2, sequence3, image1)
    mean = torch.cat(tuple(x.flatten() for x in inputs)).mean()
    std = torch.cat(tuple(x.flatten() for x in inputs)).std()
    inputs = tuple((x - mean) / std for x in inputs)
    target = None

    yield TestBaseConfig(
        model=BasicModel7_ReluMultiTensor(),
        inputs=inputs,
        target=target,
        feature_mask=feature_mask,
        baselines=tuple(torch.zeros_like(x) for x in inputs),
        frozen_features=[torch.tensor([0, 1, 2, 9, 10, 11, 18, 19, 20])],
        n_features=n_features,
    )
