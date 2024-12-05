import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from ignite.utils import convert_tensor


@dataclass
class TestBaseConfig:
    model: torch.nn.Module = None
    inputs: Tuple[torch.Tensor, ...] = None
    additional_forward_args: Tuple[torch.Tensor, ...] = None
    target: Union[torch.Tensor, List[Tuple[int, ...]]] = None
    train_baselines: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = None
    baselines: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = None
    feature_mask: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = None
    multiply_by_inputs: bool = False
    n_features: int = None
    frozen_features: List[torch.Tensor] = None
    input_layer_names: List[str] = None

    def __post_init__(self):
        assert self.model is not None, "Model must be provided"
        assert self.inputs is not None, "Inputs must be provided"

        if isinstance(self.inputs, tuple):
            for input_tensor in self.inputs:
                input_tensor.requires_grad = True
        else:
            self.inputs.requires_grad = True

    def put_to_device(self, device: str):
        self.inputs = convert_tensor(self.inputs, device=device)
        if self.target is not None:
            self.target = (
                convert_tensor(self.target, device=device)
                if isinstance(self.target, torch.Tensor)
                or (
                    isinstance(self.target, list)
                    and isinstance(self.target[0], torch.Tensor)
                )
                else self.target
            )
        if self.additional_forward_args is not None:
            if isinstance(self.additional_forward_args, tuple):
                self.additional_forward_args = tuple(
                    (
                        convert_tensor(arg, device=device)
                        if isinstance(arg, torch.Tensor)
                        else arg
                    )
                    for arg in self.additional_forward_args
                )
            else:
                self.additional_forward_args = (
                    convert_tensor(self.additional_forward_args, device=device)
                    if isinstance(self.additional_forward_args, torch.Tensor)
                    else self.additional_forward_args
                )
        if self.baselines is not None and not isinstance(self.baselines, int):
            self.baselines = convert_tensor(self.baselines, device=device)
        if self.feature_mask is not None:
            self.feature_mask = convert_tensor(self.feature_mask, device=device)
        if self.train_baselines is not None:
            self.train_baselines = convert_tensor(self.train_baselines, device=device)
        if self.frozen_features is not None:
            self.frozen_features = [
                convert_tensor(frozen_feature, device=device)
                for frozen_feature in self.frozen_features
            ]
        self.model.to(device)


@dataclasses.dataclass
class TestRuntimeConfig:
    test_name: str = ""
    target_fixture: str = None
    explainer: str = "integrated_gradients"
    explainer_kwargs: dict = None
    use_captum_explainer: bool = False
    expected: torch.Tensor = None
    delta: float = 1e-4
    override_target: Union[torch.Tensor, List[Tuple[int, ...]]] = None
    throws_exception: bool = False
    device: str = None

    def __post_init__(self):
        assert self.target_fixture is not None, "Target fixture must be provided"
        if self.explainer_kwargs is None:
            self.explainer_kwargs = {}
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.test_name is None:
            self.test_name = f"fixture={self.target_fixture}_explainer={self.explainer}"
        else:
            self.test_name = f"fixture={self.target_fixture}_explainer={self.explainer}_{self.test_name}"

    def __repr__(self):
        kws = [
            (
                f"{key}={value!r}"
                if not isinstance(value, torch.Tensor)
                else f"{key}={value.shape}"
            )
            for key, value in self.__dict__.items()
        ]
        return "{}({})".format(type(self).__name__, ", ".join(kws))
