import dataclasses

import pytest
import torch

from tests.utils.common import assert_tensor_almost_equal, grid_segmenter
from tests.utils.containers import TestBaseConfig, TestRuntimeConfig
from torchxai.explainers.factory import ExplainerFactory
from torchxai.metrics._utils.visualization import visualize_attribution
from torchxai.metrics.axiomatic.input_invariance import input_invariance


@dataclasses.dataclass
class MetricTestRuntimeConfig_(TestRuntimeConfig):
    model_type: str = "linear"
    train_and_eval_model: bool = False
    constant_shifts: torch.Tensor = None
    shifted_baselines: torch.Tensor = None
    set_baselines_to_type: str = None
    generate_feature_mask: bool = False
    visualize: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.constant_shifts is not None, "constant_shifts must be provided"
        if self.set_baselines_to_type is not None:
            assert self.set_baselines_to_type in ["zero", "black"]

    def __repr__(self):
        return super().__repr__()


@pytest.fixture
def metrics_runtime_test_configuration(request):
    runtime_config: TestRuntimeConfig = request.param
    base_config: TestBaseConfig = request.getfixturevalue(
        runtime_config.target_fixture
    )(runtime_config.model_type, runtime_config.train_and_eval_model)
    explainer = ExplainerFactory.create(
        runtime_config.explainer, base_config.model, **runtime_config.explainer_kwargs
    )
    if runtime_config.use_captum_explainer:
        explainer = explainer._explanation_fn

    if runtime_config.generate_feature_mask:
        base_config.feature_mask = grid_segmenter(base_config.inputs, 4)
    if runtime_config.set_baselines_to_type == "zero":
        base_config.baselines = 0
        runtime_config.shifted_baselines = 0
    elif runtime_config.set_baselines_to_type == "black":
        base_config.baselines = 0
        runtime_config.shifted_baselines = -1
    else:
        base_config.baselines = None
        runtime_config.shifted_baselines = None

    yield base_config, runtime_config, explainer


def setup_test_config_for_explainer(explainer, **kwargs):
    return [
        MetricTestRuntimeConfig_(
            test_name=f"{explainer}",
            target_fixture="mnist_train_configuration",
            explainer=explainer,
            train_and_eval_model=True,
            constant_shifts=torch.ones(1, 28, 28),
            use_captum_explainer=False,
            **kwargs,
        ),
        MetricTestRuntimeConfig_(
            test_name=f"captum_{explainer}",
            target_fixture="mnist_train_configuration",
            explainer=explainer,
            train_and_eval_model=True,
            constant_shifts=torch.ones(1, 28, 28),
            use_captum_explainer=True,
            **kwargs,
        ),
    ]


test_configurations = [
    # this setup is exactly the same as in the paper: https://arxiv.org/pdf/1711.00867
    # a 3-layer linear model is trained on MNIST, input invariance is computed for saliency maps
    # on 4 input samples. The expected output is [True, True, True, True]
    *setup_test_config_for_explainer(
        explainer="saliency",
        expected=torch.tensor([0.0, 0.0, 0.0, 0.0]),
    ),
    *setup_test_config_for_explainer(
        explainer="input_x_gradient",
        expected=torch.tensor(
            [0.0886, 0.0753, 0.0749, 0.0829]
        ),  # these results might not be exactly reproducible across machines
        delta=1e-3,
    ),
    # this setup is exactly the same as in the paper: https://arxiv.org/pdf/1711.00867
    # a 3-layer linear model is trained on MNIST, input invariance is computed for integrated_gradients
    # on 4 input samples. The expected output is [False, False, False, False] with zero_baseline
    *setup_test_config_for_explainer(
        explainer="integrated_gradients",
        expected=torch.tensor(
            [0.1054, 0.0862, 0.0843, 0.0868]
        ),  # these results might not be exactly reproducible across machines
        set_baselines_to_type="zero",
        explainer_kwargs={"n_steps": 200},
        delta=1e-3,
    ),
    # this setup is exactly the same as in the paper: https://arxiv.org/pdf/1711.00867
    # a 3-layer linear model is trained on MNIST, input invariance is computed for integrated_gradients
    # on 4 input samples. The expected output is [True, True, True, True] with black_baseline
    *setup_test_config_for_explainer(
        explainer="integrated_gradients",
        expected=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        set_baselines_to_type="black",
        explainer_kwargs={"n_steps": 200},
    ),
    # here apply the same logic as in the paper: https://arxiv.org/pdf/1711.00867
    # a 3-layer linear model is trained on MNIST, input invariance is computed for occlusion
    # on 4 input samples. The expected output is [True, True, True, True] with black_baseline and
    # delta=1e-3. Note that these results were not in the paper, so this shows how the implementation
    # can be used for other explainers
    *setup_test_config_for_explainer(
        explainer="occlusion",
        expected=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        set_baselines_to_type="black",
        explainer_kwargs={"sliding_window_shapes": (1, 4, 4)},
    ),
    # here apply the same logic as in the paper: https://arxiv.org/pdf/1711.00867
    # a 3-layer linear model is trained on MNIST, input invariance is computed for lime
    # on 4 input samples. The expected output is [True, True, True, True] with black_baseline and
    # delta=1e-1. Note that these results were not in the paper, so this shows how the implementation
    # can be used for other explainers
    *setup_test_config_for_explainer(
        explainer="lime",
        expected=torch.tensor(
            [0.0151, 0.0157, 0.0138, 0.0280]
        ),  # these results might not be exactly reproducible across machines
        set_baselines_to_type="black",
        explainer_kwargs={"n_samples": 200, "weight_attributions": False},
        generate_feature_mask=True,
        delta=1e-3,
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_completeness(metrics_runtime_test_configuration):
    base_config, runtime_config, explainer = metrics_runtime_test_configuration

    device = base_config.inputs.device
    kwargs = dict(target=base_config.target)
    runtime_config.constant_shifts = runtime_config.constant_shifts.to(device)
    if base_config.feature_mask is not None:
        kwargs["feature_mask"] = base_config.feature_mask.to(device)
    if base_config.baselines is not None:
        kwargs["baselines"] = (
            base_config.baselines.to(device)
            if isinstance(base_config.baselines, torch.Tensor)
            else base_config.baselines
        )
    if runtime_config.shifted_baselines is not None:
        kwargs["shifted_baselines"] = (
            runtime_config.shifted_baselines.to(device)
            if isinstance(runtime_config.shifted_baselines, torch.Tensor)
            else runtime_config.shifted_baselines
        )
    if runtime_config.use_captum_explainer:
        runtime_config.explainer_kwargs.pop(
            "weight_attributions", None
        )  # this is only available in our implementation

    output_invariance, expl_inputs, expl_shifted_inputs = input_invariance(
        explainer=explainer,
        inputs=base_config.inputs,
        constant_shifts=runtime_config.constant_shifts,
        input_layer_names=base_config.input_layer_names,
        **kwargs,
        **(
            runtime_config.explainer_kwargs
            if runtime_config.use_captum_explainer
            else {}
        ),
    )

    if runtime_config.visualize:
        # here explanations can be visualized for debugging purposes
        for input, expl_input, expl_shifted_input in zip(
            base_config.inputs, expl_inputs, expl_shifted_inputs
        ):
            visualize_attribution(input, expl_input, "Original")
            visualize_attribution(input, expl_shifted_input, "Shifted")
    assert_tensor_almost_equal(
        output_invariance.float(),
        runtime_config.expected.float(),
        delta=runtime_config.delta,
        mode="mean",
    )
