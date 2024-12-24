import inspect
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import scipy
import torch
from captum._utils.common import (
    ExpansionTypes,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    _run_forward,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor
from torchxai.metrics._utils.batching import _divide_and_aggregate_metrics
from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _format_tensor_tuple_feature_dim,
    _validate_feature_mask,
)
from torchxai.metrics._utils.perturbation import (
    _generate_random_perturbation_masks,
    default_fixed_baseline_perturb_func,
)
from torchxai.metrics.faithfulness.multi_target.faithfulness_corr import (
    _multi_target_faithfulness_corr,
)


def _faithfulness_corr(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    perturb_func: Callable = default_fixed_baseline_perturb_func(),
    n_perturb_samples: int = 10,
    max_examples_per_batch: Optional[int] = None,
    frozen_features: Optional[List[torch.Tensor]] = None,
    percent_features_perturbed: float = 0.1,
    show_progress: bool = False,
) -> Tuple[
    Union[Tensor, List[Tensor]],
    Union[Tensor, List[Tensor]],
    Union[Tensor, List[Tensor]],
]:
    def _generate_perturbations(
        current_n_perturb_samples: int,
        current_n_step: int,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
        r"""
        The perturbations are generated for each example
        `current_n_perturb_samples` times.

        For performance reasons we are not calling `perturb_func` on each example but
        on a batch that contains `current_n_perturb_samples`
        repeated instances per example.
        """

        def call_perturb_func():
            baselines_arg = None
            inputs_arg: Union[Tensor, Tuple[Tensor, ...]]
            if len(inputs_expanded) == 1:
                inputs_arg = inputs_expanded[0]
                if baselines_expanded is not None:
                    baselines_arg = cast(Tuple, baselines_expanded)[0]
                perturbation_mask_arg = perturbation_masks[0]
            else:
                inputs_arg = inputs_expanded
                baselines_arg = baselines_expanded
                perturbation_mask_arg = perturbation_masks
            pertub_kwargs = dict(
                inputs=inputs_arg,
                perturbation_masks=perturbation_mask_arg,
            )
            if (
                inspect.signature(perturb_func).parameters.get("baselines")
                and baselines_arg is not None
            ):
                pertub_kwargs["baselines"] = baselines_arg
            return perturb_func(**pertub_kwargs)

        pert_start = current_n_step - current_n_perturb_samples
        pert_end = current_n_step
        perturbation_masks = tuple(
            torch.cat(tuple(y[pert_start:pert_end] for y in x))
            for x in global_perturbation_masks
        )

        inputs_expanded = tuple(
            torch.repeat_interleave(input, current_n_perturb_samples, dim=0)
            for input in inputs
        )

        baselines_expanded = baselines
        if baselines is not None:
            baselines_expanded = tuple(
                (
                    baseline.repeat_interleave(current_n_perturb_samples, dim=0)
                    if isinstance(baseline, torch.Tensor)
                    and baseline.shape[0] == input.shape[0]
                    else baseline
                )
                for input, baseline in zip(inputs, cast(Tuple, baselines))
            )

        return call_perturb_func(), perturbation_masks

    def _validate_inputs_and_perturbations(
        inputs: Tuple[Tensor, ...],
        inputs_perturbed: Tuple[Tensor, ...],
        perturbations: Tuple[Tensor, ...],
    ) -> None:
        # asserts the sizes of the perturbations and inputs
        assert len(inputs) == len(inputs_perturbed), (
            """The number of inputs and corresponding perturbated inputs must have the same number of
            elements. Found number of inputs is: {} and inputs_perturbed:
            {}"""
        ).format(len(inputs), len(inputs_perturbed))
        # asserts the sizes of the perturbations and inputs
        assert len(perturbations) == len(inputs), (
            """The number of perturbed
            inputs and corresponding perturbations must have the same number of
            elements. Found number of inputs is: {} and perturbations:
            {}"""
        ).format(len(perturbations), len(inputs))

    def _next_faithfulness_corr_tensors(
        current_n_perturb_samples: int,
        current_n_step: int,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]:
        inputs_perturbed, perturbation_masks = _generate_perturbations(
            current_n_perturb_samples, current_n_step
        )
        inputs_perturbed = _format_tensor_into_tuples(inputs_perturbed)
        perturbation_masks = _format_tensor_into_tuples(perturbation_masks)
        # _draw_perturbated_inputs_sequences_images(inputs_perturbed)

        _validate_inputs_and_perturbations(
            cast(Tuple[Tensor, ...], inputs),
            cast(Tuple[Tensor, ...], inputs_perturbed),
            cast(Tuple[Tensor, ...], perturbation_masks),
        )

        targets_expanded = _expand_target(
            target,
            current_n_perturb_samples,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        additional_forward_args_expanded = _expand_additional_forward_args(
            additional_forward_args,
            current_n_perturb_samples,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        inputs_fwd = _run_forward(forward_func, inputs, target, additional_forward_args)
        inputs_fwd = torch.repeat_interleave(
            inputs_fwd, current_n_perturb_samples, dim=0
        )
        inputs_perturbed_fwd = _run_forward(
            forward_func,
            inputs_perturbed,
            targets_expanded,
            additional_forward_args_expanded,
        )
        perturbed_fwd_diffs = inputs_fwd - inputs_perturbed_fwd
        attributions_expanded = tuple(
            torch.repeat_interleave(attribution, current_n_perturb_samples, dim=0)
            for attribution in attributions
        )

        attributions_expanded_perturbed_sum = sum(
            tuple(
                (attribution * perturbation_mask)
                .view(attributions_expanded[0].shape[0], -1)
                .sum(dim=1)
                for attribution, perturbation_mask in zip(
                    attributions_expanded, perturbation_masks
                )
            )
        )

        # reshape to batch size dim and number of perturbations per example
        perturbed_fwd_diffs = perturbed_fwd_diffs.view(bsz, -1)
        attributions_expanded_perturbed_sum = attributions_expanded_perturbed_sum.view(
            bsz, -1
        )
        return perturbed_fwd_diffs, attributions_expanded_perturbed_sum

    def _agg_faithfulness_corr_tensors(agg_tensors, tensors):
        return tuple(
            torch.cat([agg_t, t], dim=-1) for agg_t, t in zip(agg_tensors, tensors)
        )

    with torch.no_grad():
        # perform argument formattings
        inputs = _format_tensor_into_tuples(inputs)  # type: ignore
        if baselines is not None:
            baselines = _format_baseline(baselines, cast(Tuple[Tensor, ...], inputs))
            baselines = _format_tensor_tuple_feature_dim(baselines)  # type: ignore
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        attributions = _format_tensor_into_tuples(attributions)  # type: ignore
        feature_mask = _format_tensor_into_tuples(feature_mask)  # type: ignore

        # format feature dims for single feature dim cases
        inputs = _format_tensor_tuple_feature_dim(inputs)
        attributions = _format_tensor_tuple_feature_dim(attributions)

        # Make sure that inputs and corresponding attributions have matching sizes.
        assert len(inputs) == len(attributions), (
            """The number of tensors in the inputs and
            attributions must match. Found number of tensors in the inputs is: {} and in the
            attributions: {}"""
        ).format(len(inputs), len(attributions))
        if baselines is not None:
            assert len(inputs) == len(baselines), (
                """The number of tensors in the inputs and
                baselines must match. Found number of tensors in the inputs is: {} and in the
                baselines: {}"""
            ).format(len(inputs), len(baselines))
            assert len(inputs[0]) == len(baselines[0]), (
                """The batch size in the inputs and
                baselines must match. Found batch size in the inputs is: {} and in the
                baselines: {}"""
            ).format(len(inputs[0]), len(baselines[0]))

        if feature_mask is None:
            feature_mask = _construct_default_feature_mask(attributions)

        _validate_feature_mask(feature_mask)

        # here we generate perturbation masks for the complete run in one call
        # global_perturbation_masks is a tuple of tensors, where each tensor is a perturbation mask
        # for a specific tuple input (for single inputs it is a single tensor) of shape
        # (batch_size, n_perturbations_per_sample, *input_shape)
        global_perturbation_masks = _generate_random_perturbation_masks(
            n_perturbations_per_sample=n_perturb_samples,
            feature_mask=feature_mask,
            percent_features_perturbed=percent_features_perturbed,
            frozen_features=frozen_features,
        )
        bsz = inputs[0].size(0)

        # if not normalize, directly return aggrgated MSE ((a-b)^2,)
        # else return aggregated MSE's polynomial expansion tensors (a^2, ab, b^2)
        agg_tensors = _divide_and_aggregate_metrics(
            cast(Tuple[Tensor, ...], inputs),
            n_perturb_samples,
            _next_faithfulness_corr_tensors,
            agg_func=_agg_faithfulness_corr_tensors,
            max_examples_per_batch=max_examples_per_batch,
            show_progress=show_progress,
        )
        perturbed_fwd_diffs = agg_tensors[0].detach().cpu()
        attributions_expanded_perturbed_sum = agg_tensors[1].detach().cpu()
        faithfulness_corr_scores = torch.tensor(
            [
                scipy.stats.pearsonr(x, y)[0]
                for x, y in zip(
                    attributions_expanded_perturbed_sum.numpy(),
                    perturbed_fwd_diffs.numpy(),
                )
            ]
        )
    return (
        faithfulness_corr_scores,
        attributions_expanded_perturbed_sum,
        perturbed_fwd_diffs,
    )


def faithfulness_corr(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: Union[
        List[TensorOrTupleOfTensorsGeneric], TensorOrTupleOfTensorsGeneric
    ],
    baselines: BaselineType = None,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    perturb_func: Callable = default_fixed_baseline_perturb_func(),
    n_perturb_samples: int = 10,
    max_examples_per_batch: Optional[int] = None,
    frozen_features: Optional[List[torch.Tensor]] = None,
    percent_features_perturbed: float = 0.1,
    show_progress: bool = False,
    is_multi_target: bool = False,
    return_intermediate_results: bool = False,
    return_dict: bool = False,
) -> Tuple[
    Union[Tensor, List[Tensor]],
    Union[Tensor, List[Tensor]],
    Union[Tensor, List[Tensor]],
]:
    """
    Implementation of faithfulness correlation by Bhatt et al., 2020. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library.

    The Faithfulness Correlation metric intend to capture an explanation's relative faithfulness
    (or 'fidelity') with respect to the model behaviour.

    Faithfulness correlation scores shows to what extent the predicted logits of each modified test point and
    the average explanation attribution for only the subset of features are (linearly) correlated, taking the
    average over multiple runs and test samples. The metric returns one float per input-attribution pair that
    ranges between -1 and 1, where higher scores are better.

    For each test sample, |S| features are randomly selected and replace them with baseline values (zero baseline
    or average of set). Thereafter, Pearson’s correlation coefficient between the predicted logits of each modified
    test point and the average explanation attribution for only the subset of features is calculated. Results is
    average over multiple runs and several test samples.

    In this implementation, we generate `n_perturb_samples` perturbations for each input example in the inputs tensor.
    For each perturbation we generate a random perturbation mask that selects a subset of features from the input tensor.
    We then perturb the input tensor by replacing the selected features with the corresponding baseline values.
    The subset of features to be perturbed is defined by the `feature_mask` parameter. If `feature_mask` is not provided,
    then by default a feature mask is generated for all available features in the input tensor.
    If there are feature groups defined by the feature mask, then each individual feature group is perturbed together.
    This is done by the _generate_random_perturbation_masks function, which at the start of the function generates
    all the random masks sequence.

    References:
        1) Umang Bhatt et al.: "Evaluating and aggregating feature-based model
        explanations." IJCAI (2020): 3016-3022.

    Args:
        forward_func (Callable):
                The forward function of the model or any modification of it.

        inputs (Tensor or tuple[Tensor, ...]): Input for which
                attributions are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples (aka batch size), and if
                multiple input tensors are provided, the examples must
                be aligned appropriately.

        baselines (scalar, Tensor, tuple of scalar, or Tensor):
                Baselines define reference values against which the completeness is measured which sometimes
                represent ablated values and are used to compare with the actual inputs to compute
                importance scores in attribution algorithms. They can be represented
                as:

                - a single tensor, if inputs is a single tensor, with
                  exactly the same dimensions as inputs or the first
                  dimension is one and the remaining dimensions match
                  with inputs.

                - a single scalar, if inputs is a single tensor, which will
                  be broadcasted for each input value in input tensor.

                - a tuple of tensors or scalars, the baseline corresponding
                  to each tensor in the inputs' tuple can be:

                - either a tensor with matching dimensions to
                  corresponding tensor in the inputs' tuple
                  or the first dimension is one and the remaining
                  dimensions match with the corresponding
                  input tensor.

                - or a scalar, corresponding to a tensor in the
                  inputs' tuple. This scalar value is broadcasted
                  for corresponding input tensor.

                Default: None

        attributions (Tensor or tuple[Tensor, ...]):
                Attribution scores comTensor, Tensorion algorithms can be used in local modes,
                meaning that the inputs multiplier isn't factored in the
                attribution scores.
                This can be done duing the definition of the attribution algorithm
                by passing `multipy_by_inputs=False` flag.
                For example in case of Integrated Gradients (IG) we can obtain
                local attribution scores if we define the constructor of IG as:
                ig = IntegratedGradients(multipy_by_inputs=False)

                Some attribution algorithms are inherently local.
                Examples of inherently local attribution methods include:
                Saliency, Guided GradCam, Guided Backprop and Deconvolution.

                For local attributions we can use real-valued perturbations
                whereas for global attributions that perturbation is binary.
                https://arxiv.org/abs/1901.09392

                If we want to compute the infidelity of global attributions we
                can use a binary perturbation matrix that will allow us to select
                a subset of features from `inputs` or `inputs - baselines` space.
                This will allow us to approximate sensitivity-n for a global
                attribution algorithm.

                `infidelity_perturb_func_decorator` function decorator is a helper
                function that computes perturbations under the hood if perturbed
                inputs are provided.

                For more details about how to use `infidelity_perturb_func_decorator`,
                please, read the documentation about `perturb_func`

                Attributions have the same shape and dimensionality as the inputs.
                If inputs is a single tensor then the attributions is a single
                tensor as well. If inputs is provided as a tuple of tensors
                then attributions will be tuples of tensors as well.

        feature_mask (Tensor or tuple[Tensor, ...], optional):
                    feature_mask defines a mask for the input, grouping
                    features which should be perturbed together. feature_mask
                    should contain the same number of tensors as inputs.
                    Each tensor should
                    be the same size as the corresponding input or
                    broadcastable to match the input tensor. Each tensor
                    should contain integers in the range 0 to num_features
                    - 1, and indices corresponding to the same feature should
                    have the same value.
                    Note that features within each input tensor are perturbed
                    independently (not across tensors).
                    If the forward function returns a single scalar per batch,
                    we enforce that the first dimension of each mask must be 1,
                    since attributions are returned batch-wise rather than per
                    example, so the attributions must correspond to the
                    same features (indices) in each input example.
                    If None, then a feature mask is constructed which assigns
                    each scalar within a tensor as a separate feature, which
                    is perturbed independently.
                    Default: None

        additional_forward_args (Any, optional): If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors
                or any arbitrary python types. These arguments are provided to
                forward_func in order, following the arguments in inputs.
                Note that the perturbations are not computed with respect
                to these arguments. This means that these arguments aren't
                being passed to `perturb_func` as an input argument.

                Default: None
        target (int, tuple, Tensor, or list, optional): Indices for selecting
                predictions from output(for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example, no target
                index is necessary.
                For general 2D outputs, targets can be either:

                - A single integer or a tensor containing a single
                  integer, which is applied to all input examples

                - A list of integers or a 1D tensor, with length matching
                  the number of examples in inputs (dim 0). Each integer
                  is applied as the target for the corresponding example.

                  For outputs with > 2 dimensions, targets can be either:

                - A single tuple, which contains #output_dims - 1
                  elements. This target index is applied to all examples.

                - A list of tuples with length equal to the number of
                  examples in inputs (dim 0), and each tuple containing
                  #output_dims - 1 elements. Each tuple is applied as the
                  target for the corresponding example.

                Default: None

        perturb_func (Callable):
                The perturbation function of model inputs. This function takes
                model inputs and the corresponding feature masks to be perturbed.
                Optionally it also takes baselines as input arguments and returns
                a tuple of perturbed inputs. For example:

                >>> def my_perturb_func(inputs, masks, baselines):
                >>>   <MY-LOGIC-HERE>
                >>>   return perturbed_inputs

                If there are more than one inputs passed to this function those
                will be passed to `perturb_func` as tuples in the same order as they
                are passed to the function. In addition the corresponding feature masks
                will also be passed as a tuple in the same order as inputs. See default_perturb_func
                in metrics._utils.perturbation.py for an example of a perturbation function.

                If inputs
                 - is a single tensor, the function needs to return single tensor of perturbed inputs.
                 - is a tuple of tensors, corresponding perturbed
                   inputs must be computed and returned as tuples in the
                   following format:

                   (perturbed_input1, perturbed_input2, ... perturbed_inputN)

                It is important to note that for performance reasons `perturb_func`
                isn't called for each example individually but on a batch of
                input examples that are repeated `max_examples_per_batch / batch_size`
                times within the batch.

        n_perturb_samples (int, optional): The number of times input tensors
                are perturbed. Each input example in the inputs tensor is expanded
                `n_perturb_samples`
                times before calling `perturb_func` function.

                Default: 10
        max_examples_per_batch (int, optional): The number of maximum input
                examples that are processed together. In case the number of
                examples (`input batch size * n_perturb_samples`) exceeds
                `max_examples_per_batch`, they will be sliced
                into batches of `max_examples_per_batch` examples and processed
                in a sequential order. If `max_examples_per_batch` is None, all
                examples are processed together. `max_examples_per_batch` should
                at least be equal `input batch size` and at most
                `input batch size * n_perturb_samples`.

                Default: None
        frozen_features (List[torch.Tensor], optional): A list of frozen features that are not perturbed.
                This can be useful for ignoring the input structure features like padding, etc. Default: None
                In case CLS,PAD,SEP tokens are present in the input, they can be frozen by passing the indices
                of feature masks that correspond to these tokens.
        percent_features_perturbed (float, optional): The percent_features_perturbed effectively defines
            what percentages of features in the input are perturbed in each perturbation of the input.
            Default: 0.1
        show_progress (bool, optional): Displays the progress of the metric computation.
        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class.
                Default is False.
        return_intermediate_results (bool, optional): A boolean flag that indicates whether the intermediate
                processing outputs are returned. Default is False.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
                with keys as the metric names and values as the corresponding metric outputs.
                Default is False.
        Returns:
            A tuple of three tensors:
            Tensor: - The faithfulness correlation scores of the batch. The first dimension is equal to the
                    number of examples in the input batch and the second dimension is 1.
            Tensor: - The sum of attributions for each perturbation step
                    of the batch.
            Tensor: - The forward difference between perturbed and unperturbed input for each perturbation step
                    of the batch.

    Examples::
        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> saliency = Saliency(net)
        >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> baselines = torch.zeros(2, 3, 32, 32)
        >>> # Computes saliency maps for class 3.
        >>> attribution = saliency.attribute(input, target=3)
        >>> # define a perturbation function for the input

        >>> # Computes the faithfulness correlation for saliency maps and also returns the corresponding processing
        >>> # outputs
        >>> faithfulness_corr, attribution_sums, perturbation_fwd_diffs = aopc(net, input, attribution)
    """
    metric_func = (
        _multi_target_faithfulness_corr if is_multi_target else _faithfulness_corr
    )
    (
        faithfulness_corr_scores,
        attributions_expanded_perturbed_sum,
        perturbed_fwd_diffs,
    ) = metric_func(
        forward_func=forward_func,
        inputs=inputs,
        **(
            dict(attributions_list=attributions)
            if is_multi_target
            else dict(attributions=attributions)
        ),
        baselines=baselines,
        feature_mask=feature_mask,
        additional_forward_args=additional_forward_args,
        **(dict(targets_list=target) if is_multi_target else dict(target=target)),
        perturb_func=perturb_func,
        n_perturb_samples=n_perturb_samples,
        max_examples_per_batch=max_examples_per_batch,
        frozen_features=frozen_features,
        percent_features_perturbed=percent_features_perturbed,
        show_progress=show_progress,
    )

    if return_intermediate_results:
        if return_dict:
            return {
                "faithfulness_corr_score": faithfulness_corr_scores,
                "attributions_expanded_perturbed_sum": attributions_expanded_perturbed_sum,
                "perturbed_fwd_diffs": perturbed_fwd_diffs,
            }
        else:
            return (
                faithfulness_corr_scores,
                attributions_expanded_perturbed_sum,
                perturbed_fwd_diffs,
            )
    else:
        if return_dict:
            return {"faithfulness_corr_score": faithfulness_corr_scores}
        return faithfulness_corr_scores
