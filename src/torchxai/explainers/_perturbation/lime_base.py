#!/usr/bin/env python3
import inspect
import math
import warnings
from typing import Any, Callable, List, Optional, Tuple, cast

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _reduce_list,
)
from captum._utils.models.model import Model
from captum._utils.progress import progress
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import LimeBase
from captum.log import log_usage
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from torchxai.explainers._utils import _run_forward_multi_target


class MultiTargetLimeBase(LimeBase):
    def __init__(
        self,
        forward_func: Callable,
        interpretable_model: Model,
        similarity_func: Callable,
        perturb_func: Callable,
        perturb_interpretable_space: bool,
        from_interp_rep_transform: Optional[Callable],
        to_interp_rep_transform: Optional[Callable],
    ) -> None:
        super().__init__(
            forward_func,
            interpretable_model,
            similarity_func,
            perturb_func,
            perturb_interpretable_space,
            from_interp_rep_transform,
            to_interp_rep_transform,
        )

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_samples: int = 50,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        with torch.no_grad():
            inp_tensor = (
                cast(Tensor, inputs) if isinstance(inputs, Tensor) else inputs[0]
            )
            device = inp_tensor.device

            interpretable_inps = []
            similarities = []
            outputs = []

            curr_model_inputs = []
            expanded_additional_args = None
            expanded_target = None
            perturb_generator = None
            if inspect.isgeneratorfunction(self.perturb_func):
                perturb_generator = self.perturb_func(inputs, **kwargs)

            if show_progress:
                attr_progress = progress(
                    total=math.ceil(n_samples / perturbations_per_eval),
                    desc=f"{self.get_name()} attribution",
                )
                attr_progress.update(0)

            batch_count = 0
            for _ in range(n_samples):
                if perturb_generator:
                    try:
                        curr_sample = next(perturb_generator)
                    except StopIteration:
                        warnings.warn(
                            "Generator completed prior to given n_samples iterations!"
                        )
                        break
                else:
                    curr_sample = self.perturb_func(inputs, **kwargs)
                batch_count += 1
                if self.perturb_interpretable_space:
                    interpretable_inps.append(curr_sample)
                    curr_model_inputs.append(
                        self.from_interp_rep_transform(  # type: ignore
                            curr_sample, inputs, **kwargs
                        )
                    )
                else:
                    curr_model_inputs.append(curr_sample)
                    interpretable_inps.append(
                        self.to_interp_rep_transform(  # type: ignore
                            curr_sample, inputs, **kwargs
                        )
                    )
                curr_sim = self.similarity_func(
                    inputs, curr_model_inputs[-1], interpretable_inps[-1], **kwargs
                )
                similarities.append(
                    curr_sim.flatten()
                    if isinstance(curr_sim, Tensor)
                    else torch.tensor([curr_sim], device=device)
                )

                if len(curr_model_inputs) == perturbations_per_eval:
                    if expanded_additional_args is None:
                        expanded_additional_args = _expand_additional_forward_args(
                            additional_forward_args, len(curr_model_inputs)
                        )
                    if expanded_target is None:
                        if isinstance(target, list):
                            expanded_target = [
                                _expand_target(t, len(curr_model_inputs))
                                for t in target
                            ]
                        else:
                            expanded_target = _expand_target(
                                target, len(curr_model_inputs)
                            )

                    model_out = self._evaluate_batch(
                        curr_model_inputs,
                        expanded_target,
                        expanded_additional_args,
                        device,
                    )

                    if show_progress:
                        attr_progress.update()
                    outputs.append(model_out)

                    curr_model_inputs = []

            if len(curr_model_inputs) > 0:
                expanded_additional_args = _expand_additional_forward_args(
                    additional_forward_args, len(curr_model_inputs)
                )
                if isinstance(target, list):
                    expanded_target = [
                        _expand_target(t, len(curr_model_inputs)) for t in target
                    ]
                else:
                    expanded_target = _expand_target(target, len(curr_model_inputs))

                model_out = self._evaluate_batch(
                    curr_model_inputs,
                    expanded_target,
                    expanded_additional_args,
                    device,
                )
                if show_progress:
                    attr_progress.update()
                outputs.append(model_out)

            if show_progress:
                attr_progress.close()
            combined_interp_inps = torch.cat(interpretable_inps).float()
            combined_outputs = (
                torch.cat(outputs)
                if len(outputs[0].shape) > 0
                else torch.stack(outputs)
            ).float()
            combined_sim = (
                torch.cat(similarities)
                if len(similarities[0].shape) > 0
                else torch.stack(similarities)
            ).float()

            coefs = []
            for i in range(combined_outputs.shape[1]):
                dataset = TensorDataset(
                    combined_interp_inps, combined_outputs[:, i], combined_sim
                )
                self.interpretable_model.fit(
                    DataLoader(dataset, batch_size=batch_count)
                )
                output = self.interpretable_model.representation()
                coefs.append(output)
            return coefs

    def _evaluate_batch(
        self,
        curr_model_inputs: List[TensorOrTupleOfTensorsGeneric],
        expanded_target: Tuple[TargetType, ...],
        expanded_additional_args: Any,
        device: torch.device,
    ):
        model_out = _run_forward_multi_target(
            self.forward_func,
            _reduce_list(curr_model_inputs),
            expanded_target,
            expanded_additional_args,
        )
        if isinstance(model_out, Tensor):
            if len(model_out.shape) == 1:
                return model_out.unsqueeze(-1)
            return model_out
        return torch.tensor([model_out], device=device)
