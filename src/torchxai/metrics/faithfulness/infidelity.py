#!/usr/bin/env python3

from typing import Any, Callable, Tuple, cast

import torch
from captum._utils.common import (
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    _run_forward,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from sympy import per
from torch import Tensor


def infidelity_metric(
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
    from captum.metrics import infidelity

    return infidelity(
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
