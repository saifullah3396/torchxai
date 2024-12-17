from typing import Any

import pandas as pd
import torch
from captum._utils.common import _format_tensor_into_tuples
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution

from torchxai.explainers.explainer import Explainer


class RandomExplainer(Explainer):
    """
    A Explainer class for generating random output attributions.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
    """

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the attribution function.

        Returns:
            Attribution: The initialized attribution function.
        """

        if self._is_multi_target:

            def explanation_fn(inputs, *args, **kwargs):
                inputs = _format_tensor_into_tuples(inputs)
                return [tuple(torch.randn_like(input) for input in inputs)]

            return explanation_fn

        def explanation_fn(inputs, *args, **kwargs):
            inputs = _format_tensor_into_tuples(inputs)
            return tuple(torch.randn_like(input) for input in inputs)

        return explanation_fn

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            additional_forward_args (Any): Additional arguments to forward function.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """

        return self._explanation_fn(
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            abs=False,
        )
