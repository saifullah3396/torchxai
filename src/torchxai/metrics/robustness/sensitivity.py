from typing import Any, Callable

from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor


def sensitivity_max(
    forward_func: Callable,
    perturb_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    n_perturb_samples: int = 10,
    max_examples_per_batch: int = None,
    normalize: bool = False,
) -> Tensor:
    from captum.metrics import sensitivity_max

    return sensitivity_max(
        forward_func=forward_func,
        perturb_func=perturb_func,
        inputs=inputs,
        attributions=attributions,
        baselines=baselines,
        additional_forward_args=additional_forward_args,
        target=target,
        n_perturb_samples=n_perturb_samples,
        max_examples_per_batch=max_examples_per_batch,
        normalize=normalize,
    )
