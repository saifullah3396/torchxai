import dataclasses

import pytest
import torch

from tests.utils.common import (
    assert_tensor_almost_equal,
    grid_segmenter,
    set_all_random_seeds,
)
from tests.utils.containers import TestBaseConfig, TestRuntimeConfig
from torchxai.explainers.factory import ExplainerFactory
from torchxai.metrics.axiomatic.input_invariance import input_invariance


@dataclasses.dataclass
class MetricTestRuntimeConfig(TestRuntimeConfig):
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
    if "explainer_kwargs" not in kwargs:
        kwargs["explainer_kwargs"] = {"is_multi_target": True}
    return [
        MetricTestRuntimeConfig(
            test_name=f"compare_multi_target_to_single_target",
            target_fixture="mnist_train_configuration",
            explainer=explainer,
            train_and_eval_model=True,
            constant_shifts=torch.ones(1, 28, 28),
            use_captum_explainer=False,
            override_target=[0, 1, 2],
            expected=None,
            delta=1e-8,
            **kwargs,
        ),
    ]


test_configurations = [
    *setup_test_config_for_explainer(
        explainer="saliency",
    ),
    *setup_test_config_for_explainer(
        explainer="input_x_gradient",
    ),
    *setup_test_config_for_explainer(
        explainer="integrated_gradients",
        set_baselines_to_type="zero",
        explainer_kwargs={"n_steps": 200, "is_multi_target": True},
    ),
    *setup_test_config_for_explainer(
        explainer="integrated_gradients",
        set_baselines_to_type="black",
        explainer_kwargs={"n_steps": 200, "is_multi_target": True},
    ),
    *setup_test_config_for_explainer(
        explainer="occlusion",
        set_baselines_to_type="black",
        explainer_kwargs={
            "sliding_window_shapes": (1, 4, 4),
            "strides": None,
            "is_multi_target": True,
        },
    ),
]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "metrics_runtime_test_configuration",
    test_configurations,
    ids=[f"{idx}_{config.test_name}" for idx, config in enumerate(test_configurations)],
    indirect=True,
)
def test_input_invariance(metrics_runtime_test_configuration):
    base_config, runtime_config, explainer = metrics_runtime_test_configuration

    device = base_config.inputs.device
    kwargs = dict()
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

    set_all_random_seeds(1234)
    output_invariances_1, expl_inputs_1, expl_shifted_inputs_1 = input_invariance(
        explainer=explainer,
        inputs=base_config.inputs,
        constant_shifts=runtime_config.constant_shifts,
        input_layer_names=base_config.input_layer_names,
        target=runtime_config.override_target,
        **kwargs,
        **(
            runtime_config.explainer_kwargs
            if runtime_config.use_captum_explainer
            else {}
        ),
        is_multi_target=True,
    )

    output_invariances_2 = []
    expl_inputs_2 = []
    expl_shifted_inputs_2 = []

    explainer.is_multi_target = False
    for target in runtime_config.override_target:
        set_all_random_seeds(1234)
        output_invariance, expl_inputs, expl_shifted_inputs = input_invariance(
            explainer=explainer,
            inputs=base_config.inputs,
            constant_shifts=runtime_config.constant_shifts,
            input_layer_names=base_config.input_layer_names,
            target=target,
            **kwargs,
            **(
                runtime_config.explainer_kwargs
                if runtime_config.use_captum_explainer
                else {}
            ),
        )
        output_invariances_2.append(output_invariance)
        expl_inputs_2.append(expl_inputs)
        expl_shifted_inputs_2.append(expl_shifted_inputs)

    for x, y in zip(output_invariances_1, output_invariances_2):
        assert_tensor_almost_equal(
            x.float(), y.float(), delta=runtime_config.delta, mode="mean"
        )

    for x, y in zip(expl_inputs_1, expl_inputs_2):
        assert_tensor_almost_equal(
            x.float(), y.float(), delta=runtime_config.delta, mode="mean"
        )

    for x, y in zip(expl_shifted_inputs_1, expl_shifted_inputs_2):
        assert_tensor_almost_equal(
            x.float(), y.float(), delta=runtime_config.delta, mode="mean"
        )
