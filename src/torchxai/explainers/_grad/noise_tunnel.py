#!/usr/bin/env python3
from enum import Enum
from typing import Any, List, Tuple, Union, cast

import torch
from captum._utils.common import (
    _expand_and_update_additional_forward_args,
    _expand_and_update_baselines,
    _expand_and_update_feature_mask,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._utils.common import _validate_noise_tunnel_type
from captum.log import log_usage
from torch import Tensor

from torchxai.explainers._utils import _expand_and_update_target_multi_target


class NoiseTunnelType(Enum):
    smoothgrad = 1
    smoothgrad_sq = 2
    vargrad = 3


from captum.attr import NoiseTunnel

SUPPORTED_NOISE_TUNNEL_TYPES = list(NoiseTunnelType.__members__.keys())


class MultiTargetNoiseTunnel(NoiseTunnel):
    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        nt_type: str = "smoothgrad",
        nt_samples: int = 5,
        nt_samples_batch_size: int = None,
        stdevs: Union[float, Tuple[float, ...]] = 1.0,
        draw_baseline_from_distrib: bool = False,
        **kwargs: Any,
    ) -> Union[
        Union[
            Tensor,
            Tuple[Tensor, Tensor],
            Tuple[Tensor, ...],
            Tuple[Tuple[Tensor, ...], Tensor],
        ]
    ]:
        def add_noise_to_inputs(nt_samples_partition: int) -> Tuple[Tensor, ...]:
            if isinstance(stdevs, tuple):
                assert len(stdevs) == len(inputs), (
                    "The number of input tensors "
                    "in {} must be equal to the number of stdevs values {}".format(
                        len(inputs), len(stdevs)
                    )
                )
            else:
                assert isinstance(
                    stdevs, float
                ), "stdevs must be type float. " "Given: {}".format(type(stdevs))
                stdevs_ = (stdevs,) * len(inputs)
            return tuple(
                (
                    add_noise_to_input(
                        input, stdev, nt_samples_partition
                    ).requires_grad_()
                    if self.is_gradient_method
                    else add_noise_to_input(input, stdev, nt_samples_partition)
                )
                for (input, stdev) in zip(inputs, stdevs_)
            )

        def add_noise_to_input(
            input: Tensor, stdev: float, nt_samples_partition: int
        ) -> Tensor:
            # batch size
            bsz = input.shape[0]

            # expand input size by the number of drawn samples
            input_expanded_size = (bsz * nt_samples_partition,) + input.shape[1:]

            # expand stdev for the shape of the input and number of drawn samples
            stdev_expanded = torch.tensor(stdev, device=input.device).repeat(
                input_expanded_size
            )

            # draws `np.prod(input_expanded_size)` samples from normal distribution
            # with given input parametrization
            # FIXME it look like it is very difficult to make torch.normal
            # deterministic this needs an investigation
            noise = torch.normal(0, stdev_expanded)
            return input.repeat_interleave(nt_samples_partition, dim=0) + noise

        def update_sum_attribution_and_sq(
            sum_attribution: List[Tensor],
            sum_attribution_sq: List[Tensor],
            attribution: Tensor,
            i: int,
            j: int,
            nt_samples_batch_size_inter: int,
        ) -> None:
            bsz = attribution.shape[0] // nt_samples_batch_size_inter
            attribution_shape = cast(
                Tuple[int, ...], (bsz, nt_samples_batch_size_inter)
            )
            if len(attribution.shape) > 1:
                attribution_shape += cast(Tuple[int, ...], tuple(attribution.shape[1:]))

            attribution = attribution.view(attribution_shape)
            current_attribution_sum = attribution.sum(dim=1, keepdim=False)
            current_attribution_sq = torch.sum(attribution**2, dim=1, keepdim=False)

            sum_attribution[i][j] = (
                current_attribution_sum
                if not isinstance(sum_attribution[i][j], torch.Tensor)
                else sum_attribution[i][j] + current_attribution_sum
            )
            sum_attribution_sq[i][j] = (
                current_attribution_sq
                if not isinstance(sum_attribution_sq[i][j], torch.Tensor)
                else sum_attribution_sq[i][j] + current_attribution_sq
            )

        def compute_partial_attribution(
            inputs_with_noise_partition: Tuple[Tensor, ...], kwargs_partition: Any
        ) -> Tuple[Tuple[Tensor, ...], bool, Union[None, Tensor]]:
            # smoothgrad_Attr(x) = 1 / n * sum(Attr(x + N(0, sigma^2))
            # NOTE: using __wrapped__ such that it does not log the inner logs

            attributions = attr_func.__wrapped__(  # type: ignore
                self.attribution_method,  # self
                (
                    inputs_with_noise_partition
                    if is_inputs_tuple
                    else inputs_with_noise_partition[0]
                ),
                **kwargs_partition,
            )
            delta = None

            if self.is_delta_supported and return_convergence_delta:
                attributions, delta = attributions

            is_attrib_tuple = [
                _is_tuple(attribution_per_target)
                for attribution_per_target in attributions
            ]
            attributions = [
                _format_tensor_into_tuples(attribution_per_target)
                for attribution_per_target in attributions
            ]

            return (
                cast(List[Tuple[Tensor, ...]], attributions),
                cast(List[bool], is_attrib_tuple),
                delta,
            )

        def expand_partial(nt_samples_partition: int, kwargs_partial: dict) -> None:
            # if the algorithm supports targets, baselines and/or
            # additional_forward_args they will be expanded based
            # on the nt_samples_partition and corresponding kwargs
            # variables will be updated accordingly
            _expand_and_update_additional_forward_args(
                nt_samples_partition, kwargs_partial
            )
            _expand_and_update_target_multi_target(nt_samples_partition, kwargs_partial)
            _expand_and_update_baselines(
                cast(Tuple[Tensor, ...], inputs),
                nt_samples_partition,
                kwargs_partial,
                draw_baseline_from_distrib=draw_baseline_from_distrib,
            )
            _expand_and_update_feature_mask(nt_samples_partition, kwargs_partial)

        def compute_smoothing(
            expected_attributions: Tuple[Union[Tensor], ...],
            expected_attributions_sq: Tuple[Union[Tensor], ...],
        ) -> Tuple[Tensor, ...]:
            if NoiseTunnelType[nt_type] == NoiseTunnelType.smoothgrad:
                return expected_attributions

            if NoiseTunnelType[nt_type] == NoiseTunnelType.smoothgrad_sq:
                return expected_attributions_sq

            vargrad = tuple(
                expected_attribution_sq - expected_attribution * expected_attribution
                for expected_attribution, expected_attribution_sq in zip(
                    expected_attributions, expected_attributions_sq
                )
            )

            return cast(Tuple[Tensor, ...], vargrad)

        def update_partial_attribution_and_delta(
            multi_target_attributions_partial: Tuple[Tensor, ...],
            multi_target_delta_partial: Tensor,
            multi_target_sum_attributions: List[Tensor],
            multi_target_sum_attributions_sq: List[Tensor],
            multi_target_delta_partial_list: List[Tensor],
            nt_samples_partial: int,
        ) -> None:
            for i, attributions_partial in enumerate(multi_target_attributions_partial):
                for j, attribution_partial in enumerate(attributions_partial):
                    update_sum_attribution_and_sq(
                        multi_target_sum_attributions,
                        multi_target_sum_attributions_sq,
                        attribution_partial,
                        i,
                        j,
                        nt_samples_partial,
                    )
            if self.is_delta_supported and return_convergence_delta:
                multi_target_delta_partial_list.append(multi_target_delta_partial)

        return_convergence_delta: bool
        return_convergence_delta = (
            "return_convergence_delta" in kwargs and kwargs["return_convergence_delta"]
        )
        with torch.no_grad():
            nt_samples_batch_size = (
                nt_samples
                if nt_samples_batch_size is None
                else min(nt_samples, nt_samples_batch_size)
            )

            nt_samples_partition = nt_samples // nt_samples_batch_size

            # Keeps track whether original input is a tuple or not before
            # converting it into a tuple.
            is_inputs_tuple = isinstance(inputs, tuple)

            inputs = _format_tensor_into_tuples(inputs)  # type: ignore

            _validate_noise_tunnel_type(nt_type, SUPPORTED_NOISE_TUNNEL_TYPES)

            kwargs_copy = kwargs.copy()
            expand_partial(nt_samples_batch_size, kwargs_copy)

            attr_func = self.attribution_method.attribute

            multi_target_sum_attributions: List[List[Union[None, Tensor]]] = []
            multi_target_sum_attributions_sq: List[List[Union[None, Tensor]]] = []
            multi_target_delta_partial_list: List[List[Tensor]] = []

            for _ in range(nt_samples_partition):
                inputs_with_noise = add_noise_to_inputs(nt_samples_batch_size)
                (
                    multi_target_attributions_partial,
                    multi_target_is_attrib_tuple,
                    multi_target_delta_partial,
                ) = compute_partial_attribution(inputs_with_noise, kwargs_copy)

                if len(multi_target_sum_attributions) == 0:
                    multi_target_sum_attributions = [[]] * len(
                        multi_target_attributions_partial
                    )
                    multi_target_sum_attributions_sq = [[]] * len(
                        multi_target_attributions_partial
                    )
                    multi_target_delta_partial_list = [[]] * len(
                        multi_target_attributions_partial
                    )

                    for target_index in range(len(multi_target_attributions_partial)):
                        multi_target_sum_attributions[target_index] = [None] * len(
                            multi_target_attributions_partial[target_index]
                        )
                        multi_target_sum_attributions_sq[target_index] = [None] * len(
                            multi_target_attributions_partial[target_index]
                        )

                update_partial_attribution_and_delta(
                    multi_target_attributions_partial,
                    multi_target_delta_partial,
                    multi_target_sum_attributions,
                    multi_target_sum_attributions_sq,
                    multi_target_delta_partial_list,
                    nt_samples_batch_size,
                )

            nt_samples_remaining = (
                nt_samples - nt_samples_partition * nt_samples_batch_size
            )
            if nt_samples_remaining > 0:
                inputs_with_noise = add_noise_to_inputs(nt_samples_remaining)
                expand_partial(nt_samples_remaining, kwargs)
                (
                    multi_target_attributions_partial,
                    multi_target_is_attrib_tuple,
                    multi_target_delta_partial,
                ) = compute_partial_attribution(inputs_with_noise, kwargs)

                update_partial_attribution_and_delta(
                    multi_target_attributions_partial,
                    multi_target_delta_partial,
                    multi_target_sum_attributions,
                    multi_target_sum_attributions_sq,
                    multi_target_delta_partial_list,
                    nt_samples_remaining,
                )

            multi_target_expected_attributions = [
                tuple(
                    [
                        cast(Tensor, sum_attribution) * 1 / nt_samples
                        for sum_attribution in sum_attributions
                    ]
                )
                for sum_attributions in multi_target_sum_attributions
            ]
            multi_target_expected_attributions_sq = [
                tuple(
                    [
                        cast(Tensor, sum_attribution_sq) * 1 / nt_samples
                        for sum_attribution_sq in sum_attributions_sq
                    ]
                )
                for sum_attributions_sq in multi_target_sum_attributions_sq
            ]
            multi_target_attributions = [
                compute_smoothing(
                    cast(Tuple[Tensor, ...], expected_attributions),
                    cast(Tuple[Tensor, ...], expected_attributions_sq),
                )
                for expected_attributions, expected_attributions_sq in zip(
                    multi_target_expected_attributions,
                    multi_target_expected_attributions_sq,
                )
            ]

            delta = None
            if self.is_delta_supported and return_convergence_delta:
                delta = [
                    torch.cat(delta_partial_list, dim=0)
                    for delta_partial_list in multi_target_delta_partial_list
                ]

        return self._apply_checks_and_return_attributions(
            multi_target_attributions,
            multi_target_is_attrib_tuple,
            return_convergence_delta,
            delta,
        )

    def _apply_checks_and_return_attributions(
        self,
        multi_target_attributions: Tuple[Tensor, ...],
        is_attrib_tuple: bool,
        return_convergence_delta: bool,
        delta: Union[None, Tensor],
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        multi_target_attributions = [
            _format_output(is_attrib_tuple, attributions)
            for attributions in multi_target_attributions
        ]

        ret = (
            (multi_target_attributions, cast(List[Tensor], delta))
            if self.is_delta_supported and return_convergence_delta
            else multi_target_attributions
        )
        ret = cast(
            Union[
                List[TensorOrTupleOfTensorsGeneric],
                Tuple[List[TensorOrTupleOfTensorsGeneric], List[Tensor]],
            ],
            ret,
        )
        return ret

    def has_convergence_delta(self) -> bool:
        return self.is_delta_supported
