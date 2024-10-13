import copy
import dataclasses

import pytest  # noqa

from tests.utils.common import (
    compare_explanation_per_target,
    compute_explanations,
    set_all_random_seeds,
)
from tests.utils.containers import TestRuntimeConfig
from torchxai.explainers.factory import ExplainerFactory


@dataclasses.dataclass
class ExplainersTestRuntimeConfig(TestRuntimeConfig):
    is_multi_target: bool = False


def make_config_for_explainer(
    target_fixture,
    explainer,
    test_name_suffix="",
    config_class=ExplainersTestRuntimeConfig,
    **kwargs,
):
    return config_class(
        test_name=f"{target_fixture}_{explainer}{test_name_suffix}",
        target_fixture=target_fixture,
        explainer=explainer,
        **kwargs,
    )


def make_config_for_explainers_with_internal_batch_size(
    *args,
    **kwargs,
):
    internal_batch_sizes = kwargs.pop("internal_batch_sizes", [None])
    return [
        make_config_for_explainer(
            *args,
            **kwargs,
            explainer_kwargs={"internal_batch_size": internal_batch_size},
            test_name_suffix=f"_internal_batch_size_{internal_batch_size}",
        )
        for internal_batch_size in internal_batch_sizes
    ]


def run_explainer_test_with_config(base_config, runtime_config):
    if not isinstance(runtime_config.expected, list):
        runtime_config.expected = [runtime_config.expected]
    if not isinstance(base_config.target, list):
        base_config.target = [base_config.target]

    assert len(base_config.target) == len(
        runtime_config.expected
    ), "The number of targets must be equal to the number of expected outputs"

    single_target_explanations = []
    for curr_target, curr_expected in zip(base_config.target, runtime_config.expected):
        curr_runtime_config = copy.deepcopy(runtime_config)
        curr_base_config = copy.deepcopy(base_config)

        # run for normal case
        curr_runtime_config.expected = curr_expected
        curr_base_config.target = curr_target

        # get explanation using single-target explainer
        explanations = run_single_test(curr_base_config, curr_runtime_config)

        curr_runtime_config.expected = [curr_expected]
        multi_target_explanations_1 = None
        if curr_target is None or isinstance(curr_target, int):
            # if expected is integer target then we can run this for both as batch and single input so test for both
            multi_target_explanations_1 = run_single_test(
                curr_base_config, curr_runtime_config, is_multi_target=True
            )

        # get explanation using multi-target explainer but as list input and verify the output is same as single target
        if curr_base_config.target is not None:
            curr_base_config.target = [curr_base_config.target]

        multi_target_explanations_2 = run_single_test(
            curr_base_config, curr_runtime_config, is_multi_target=True
        )
        if multi_target_explanations_1 is not None:
            for m1, m2 in zip(multi_target_explanations_1, multi_target_explanations_2):
                compare_explanation_per_target(m1, m2, delta=runtime_config.delta)

        if multi_target_explanations_2 is None:
            assert explanations is None
        else:
            compare_explanation_per_target(
                multi_target_explanations_2[0], explanations, delta=runtime_config.delta
            )

        single_target_explanations.append(explanations)

    if len(single_target_explanations) > 1:
        multi_target_explanations = run_single_test(
            base_config, runtime_config, is_multi_target=True
        )

        for multi_target_explanation, single_target_explanation in zip(
            multi_target_explanations, single_target_explanations
        ):
            # target explanation in the list should match the single target explanations at the same index
            compare_explanation_per_target(
                multi_target_explanation,
                single_target_explanation,
                delta=runtime_config.delta,
            )


def run_single_test(base_config, runtime_config, is_multi_target=False):
    set_all_random_seeds(1234)

    if is_multi_target:
        runtime_config.explainer_kwargs["is_multi_target"] = True

    base_config.model.to(base_config.device)
    explainer = ExplainerFactory.create(
        runtime_config.explainer, base_config.model, **runtime_config.explainer_kwargs
    )
    if runtime_config.throws_exception:
        with pytest.raises(Exception) as e_info:
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
                device=base_config.device,
                **runtime_config.explainer_kwargs,
            )
        return

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
        device=base_config.device,
        **runtime_config.explainer_kwargs,
    )

    has_expected = (
        runtime_config.expected is not None
        if not isinstance(runtime_config.expected, list)
        else all(v is not None for v in runtime_config.expected)
    )
    if has_expected:
        if is_multi_target:
            assert isinstance(
                explanations, list
            ), "The output explanations must be a list when is_multi_target is True"
            assert isinstance(
                runtime_config.expected, list
            ), "The expected explanations must be a list when is_multi_target is True"
            assert len(explanations) == len(runtime_config.expected), (
                "The number of output explanations must be equal to the number of expected outputs "
                "when is_multi_target is True"
            )

            for output_per_target, expected_per_target in zip(
                explanations, runtime_config.expected
            ):
                compare_explanation_per_target(
                    output_per_target, expected_per_target, delta=runtime_config.delta
                )
        else:
            compare_explanation_per_target(
                explanations, runtime_config.expected, delta=runtime_config.delta
            )
    return explanations
