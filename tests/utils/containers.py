import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch


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
    input_layer_names: List[str] = None
    device: str = None

    def __post_init__(self):
        assert self.model is not None, "Model must be provided"
        assert self.inputs is not None, "Inputs must be provided"

        if isinstance(self.inputs, tuple):
            for input_tensor in self.inputs:
                input_tensor.requires_grad = True
        else:
            self.inputs.requires_grad = True

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclasses.dataclass
class TestRuntimeConfig:
    test_name: str
    target_fixture: str = None
    explainer: str = "integrated_gradients"
    explainer_kwargs: dict = None
    use_captum_explainer: bool = False
    expected: torch.Tensor = None
    delta: float = 1e-4
    override_target: Union[torch.Tensor, List[Tuple[int, ...]]] = None
    throws_exception: bool = False

    def __post_init__(self):
        assert self.target_fixture is not None, "Target fixture must be provided"
        if self.explainer_kwargs is None:
            self.explainer_kwargs = {}

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
