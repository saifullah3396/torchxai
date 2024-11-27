#!/usr/bin/env python3

from typing import Any, Callable, List, Optional, Tuple, Union, cast

import numpy as np
import scipy
import torch
import tqdm
from captum._utils.common import (
    ExpansionTypes,
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_tensor_into_tuples,
    _run_forward,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor

from torchxai.metrics._utils.batching import (
    _divide_and_aggregate_metrics_n_perturbations_per_feature,
)
from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _feature_mask_to_perturbation_mask,
    _reduce_tensor_with_indices_non_deterministic,
    _split_tensors_to_tuple_tensors,
    _tuple_tensors_to_tensors,
    _validate_feature_mask,
)
from torchxai.metrics._utils.perturbation import default_random_perturb_func
from torchxai.metrics.axiomatic.multi_target.monotonicity_corr_and_non_sens import (
    _multi_target_monotonicity_corr_and_non_sens,
)


def eval_monotonicity_corr_and_non_sens_single_sample(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    n_perturbations_per_feature: int = 10,
    additional_forward_args: Any = None,
    target: TargetType = None,
    perturb_func: Callable = default_random_perturb_func(),
    max_features_processed_per_batch: int = None,
    frozen_features: Optional[List[torch.Tensor]] = None,
    use_absolute_attributions: bool = True,
    eps: float = 1e-5,
    show_progress: bool = False,
    return_ratio: bool = True,
) -> Tensor:
    def _generate_perturbations(
        current_n_perturbed_features: int, current_perturbation_mask: Tensor
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
        r"""
        The perturbations are generated for each example
        `current_n_perturbed_features * n_perturbations_per_feature` times.

        For performance reasons we are not calling `perturb_func` on each example but
        on a batch that contains `current_n_perturbed_features * n_perturbations_per_feature`
        repeated instances per example.
        """

        def call_perturb_func():
            r""" """
            inputs_pert: Union[Tensor, Tuple[Tensor, ...]]
            if len(inputs_expanded) == 1:
                inputs_pert = inputs_expanded[0]
                perturbation_masks = perturbation_mask_expanded[0]
            else:
                inputs_pert = inputs_expanded
                perturbation_masks = perturbation_mask_expanded
            return perturb_func(
                inputs=inputs_pert, perturbation_masks=perturbation_masks
            )

        # repeat each current_n_perturbed_features times
        inputs_expanded = tuple(
            input.repeat(
                n_perturbations_per_feature * current_n_perturbed_features,
                *tuple([1] * len(input.shape[1:])),
            )
            for input in inputs
        )

        # repeat each perturbation mask n_perturbations_per_feature times
        perturbation_mask_expanded = current_perturbation_mask.repeat_interleave(
            repeats=n_perturbations_per_feature, dim=0
        )

        # split back to tuple tensors
        perturbation_mask_expanded = _split_tensors_to_tuple_tensors(
            perturbation_mask_expanded, flattened_mask_shape
        )

        # view as input shape (this is only necessary for edge cases where input is of (1, 1) shape)
        perturbation_mask_expanded = tuple(
            mask.view_as(input)
            for mask, input in zip(perturbation_mask_expanded, inputs_expanded)
        )
        return call_perturb_func()

    def _validate_inputs_and_perturbations(
        inputs: Tuple[Tensor, ...],
        inputs_perturbed: Tuple[Tensor, ...],
    ) -> None:
        # asserts the sizes of the perturbations and inputs
        assert len(inputs_perturbed) == len(inputs), (
            """The number of perturbed
            inputs and corresponding perturbations must have the same number of
            elements. Found number of inputs is: {} and perturbations:
            {}"""
        ).format(len(inputs_perturbed), len(inputs))

        # asserts the shapes of the perturbations and perturbed inputs
        for inputs, input_perturbed in zip(inputs, inputs_perturbed):
            assert inputs[0].shape == input_perturbed[0].shape, (
                """Perturbed input
                and corresponding perturbation must have the same shape and
                dimensionality. Found perturbation shape is: {} and the input shape
                is: {}"""
            ).format(inputs[0].shape, input_perturbed[0].shape)

    def _next_monotonicity_corr_tensors(
        current_n_perturbed_features: int,
        current_n_steps: int,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]:
        current_feature_indices = torch.arange(
            current_n_steps - current_n_perturbed_features,
            current_n_steps,
            device=inputs[0].device,
        )
        current_feature_indices = torch.arange(
            current_n_steps - current_n_perturbed_features, current_n_steps
        )
        current_perturbation_mask = global_perturbation_masks[current_feature_indices]
        inputs_perturbed = _generate_perturbations(
            current_n_perturbed_features, current_perturbation_mask
        )
        inputs_perturbed = _format_tensor_into_tuples(inputs_perturbed)
        _validate_inputs_and_perturbations(
            cast(Tuple[Tensor, ...], inputs),
            cast(Tuple[Tensor, ...], inputs_perturbed),
        )
        additional_forward_args_expanded = _expand_additional_forward_args(
            additional_forward_args,
            n_perturbations_per_feature * current_n_perturbed_features,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        targets_expanded = _expand_target(
            target,
            current_n_perturbed_features,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        inputs_fwd = _run_forward(forward_func, inputs, target, additional_forward_args)
        inputs_fwd_inv = (
            1.0 if torch.abs(inputs_fwd) < eps else 1.0 / torch.abs(inputs_fwd)
        )
        inputs_perturbed_fwd = _run_forward(
            forward_func,
            inputs_perturbed,
            targets_expanded,
            additional_forward_args_expanded,
        )
        perturbed_fwd_diffs = inputs_perturbed_fwd - inputs_fwd
        perturbed_fwd_diffs = perturbed_fwd_diffs.chunk(current_n_perturbed_features)

        # each element in the tuple corresponds to a single feature group
        curr_perturbed_fwd_diffs_relative_vars = tuple(
            torch.mean(x**2) * (inputs_fwd_inv**2) for x in perturbed_fwd_diffs
        )

        # return the perturbed forward differences and the current feature attribution scores
        curr_perturbed_fwd_diffs_relative_vars = tuple(
            x.item() for x in curr_perturbed_fwd_diffs_relative_vars
        )
        curr_feature_group_attribution_scores = tuple(
            x.item() for x in reduced_attributions[current_feature_indices]
        )
        return list(curr_perturbed_fwd_diffs_relative_vars), list(
            curr_feature_group_attribution_scores
        )

    def _agg_monotonicity_corr_tensors(agg_tensors, tensors):
        return tuple(agg_t + t for agg_t, t in zip(agg_tensors, tensors))

    with torch.no_grad():
        bsz = inputs[0].size(0)
        assert bsz == 1, "Batch size must be 1 for monotonicity_corr_single_sample"
        if feature_mask is None:
            feature_mask = _construct_default_feature_mask(attributions)

        # flatten the feature mask
        feature_mask_flattened, flattened_mask_shape = _tuple_tensors_to_tensors(
            feature_mask
        )

        # validate feature masks are increasing non-negative
        _validate_feature_mask(feature_mask_flattened)

        # flatten all attributions in the input, this must be done after the feature masks are flattened as
        # feature masks may depened on attribution
        attributions, _ = _tuple_tensors_to_tensors(attributions)
        reduced_attributions, _ = _reduce_tensor_with_indices_non_deterministic(
            attributions[0], indices=feature_mask_flattened[0]
        )

        if use_absolute_attributions:
            reduced_attributions = reduced_attributions.abs()

        # this assumes a batch size of 1, this will not work for batch size > 1
        feature_indices = feature_mask_flattened.unique()
        global_perturbation_masks = _feature_mask_to_perturbation_mask(
            feature_mask_flattened, feature_indices, frozen_features
        )
        n_features = global_perturbation_masks.shape[0]

        # for idx, perturbation_mask in enumerate(global_perturbation_masks):
        #     if idx % 10 == 0:

        #         m1 = perturbation_mask[: 512 * 768].view(512, 768)[:, 0]
        #         m2 = perturbation_mask[512 * 768 : 2 * 512 * 768].view(512, 768)[:, 0]
        #         m3 = perturbation_mask[2 * 512 * 768 : 3 * 512 * 768].view(512, 768)[
        #             :, 0
        #         ]
        #         m4 = (
        #             perturbation_mask[3 * 512 * 768 :]
        #             .view(3, 224, 224)
        #             .permute(1, 2, 0)
        #         )
        #         import matplotlib.pyplot as plt

        #         fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        #         ax[0].plot(m1.cpu().numpy())
        #         ax[1].plot(m2.cpu().numpy())
        #         ax[2].plot(m3.cpu().numpy())
        #         ax[3].imshow(m4.cpu().float().numpy())
        #         plt.show()
        #     # m1 = global_perturbation_masks[]
        # exit()

        # the logic for this implementation as as follows:
        # each input is repeated n_perturbations_per_feature times to create an input of shape
        # tuple(
        #   [(n_perturbations_per_feature, input.shape[0], ...), (n_perturbations_per_feature, input.shape[0], ...)]
        # )
        # then in each perturbation step, every feature is sequentually perturbed n_perturbations_per_feature times
        # so perturbation step will have the n_pertrubations_per_feature perturbations for the first feature, then
        # n_perturbations_per_feature perturbations for the second feature and so on.
        # and the total perturbation steps n_features are equal to total features that need to be perturbed
        # in case of feature_mask, n_features is the total number of feature groups present in the inputs
        # each group is perturbed together n_perturbations_per_feature times
        # in case there is no feature masks, then a corresponding feature mask is generated for each input feature
        agg_tensors = _divide_and_aggregate_metrics_n_perturbations_per_feature(
            n_perturbations_per_feature,
            global_perturbation_masks.shape[0],
            _next_monotonicity_corr_tensors,
            agg_func=_agg_monotonicity_corr_tensors,
            max_features_processed_per_batch=max_features_processed_per_batch,
            show_progress=show_progress,
        )

        # compute monotonocity corr metric
        def compute_monotonocity_corr(
            perturbed_fwd_diffs_relative_vars: np.ndarray,
            feature_group_attribution_scores: np.ndarray,
        ):
            # find the spearman corr between the attribution scores and the model forward variances
            # this corr should be close to 1 if the model forward variances are monotonically increasing with the attribution scores
            # this means that features that have a lower attribution score are directly correlated with lower effect on the model output
            return scipy.stats.spearmanr(
                np.abs(feature_group_attribution_scores),
                perturbed_fwd_diffs_relative_vars,
            )[0]

        # compute non-sensitivity metric
        def compute_non_sens(
            perturbed_fwd_diffs_relative_vars: np.ndarray,
            feature_group_attribution_scores: np.ndarray,
        ):
            # find the indices of features that have a zero attribution score, every attribution score value less
            # than non_sens_eps is considered zero
            zero_attribution_features = set(
                list(
                    np.argwhere(
                        np.abs(feature_group_attribution_scores) < eps
                    ).flatten()
                )
            )
            zero_variance_features = set(
                list(
                    np.argwhere(
                        np.abs(perturbed_fwd_diffs_relative_vars) < eps
                    ).flatten()
                )
            )

            # find the symmetric difference of the zero attribution features and the zero variance features
            # this set should be empty if the model is non-sensitive to the zero attribution features
            # symmetric difference will give the oppposite of the intersection of the two sets
            # therefore non-sensitivity corresponds to the number of features that have either:
            # 1. zero attribution scores and non-zero model forward variances
            # 2. non-zero attribution scores and zero model forward variances
            # a higher non-sensitivity score indicates that the model is more sensitive to the zero attribution features
            # and a lower non-sensitivity score indicates that the model is non-sensitive to the zero attribution features
            non_sens = len(
                zero_attribution_features.symmetric_difference(zero_variance_features)
            )
            if return_ratio:
                return non_sens / n_features
            return non_sens

        agg_tensors = tuple(np.array(x) for x in agg_tensors)
        perturbed_fwd_diffs_relative_vars = agg_tensors[0]
        feature_group_attribution_scores = agg_tensors[1]

        monotonicity_corr = compute_monotonocity_corr(
            perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores
        )
        non_sens = compute_non_sens(
            perturbed_fwd_diffs_relative_vars, feature_group_attribution_scores
        )
    return (
        monotonicity_corr,
        non_sens,
        n_features,
        perturbed_fwd_diffs_relative_vars,
        feature_group_attribution_scores,
    )


def monotonicity_corr_and_non_sens(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    perturb_func: Callable = default_random_perturb_func(),
    n_perturbations_per_feature: int = 10,
    max_features_processed_per_batch: int = None,
    eps: float = 1e-5,
    is_multi_target: bool = False,
    frozen_features: Optional[List[torch.Tensor]] = None,
    use_absolute_attributions: bool = True,
    show_progress: bool = False,
    return_intermediate_results: bool = False,
    return_dict: bool = False,
    return_ratio: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Implementation of Monotonicity Correlation and NonSensitivity by Nguyen at el., 2020. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library. Both of these implementations are combined together since the monotonicity and non-sensitivity
    metric are closely related and can be computed together.

    1. Monotonicity measures the (Spearman’s) correlation coefficient of the absolute values of the attributions
    and the uncertainty in probability estimation. The paper argues that if attributions are not monotonic
    then they are not providing the correct importance of the feature.

    2. Non-sensitivity measures if zero-importance is only assigned to features, that the model is not
    functionally dependent on.

    References:
        1) An-phi Nguyen and María Rodríguez Martínez.: "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).
        2) Marco Ancona et al.: "Explaining Deep Neural Networks with a Polynomial Time Algorithm for Shapley
        Values Approximation." ICML (2019): 272-281.
        3) Grégoire Montavon et al.: "Methods for interpreting and
        understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

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
                Attribution scores computed based on an attribution algorithm.
                This attribution scores can be computed using the implementations
                provided in the `captum.attr` package. Some of those attribution
                approaches are so called global methods, which means that
                they factor in model inputs' multiplier, as described in:
                https://arxiv.org/abs/1711.06104
                Many global attribution algorithms can be used in local modes,
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
                input examples that are repeated `max_features_processed_per_batch / batch_size`
                times within the batch.

        n_perturbations_per_feature (int, optional): The number of times each feature is perturbed.
                Each input example in the inputs tensor is expanded `n_perturbations_per_feature`
                times before calling `perturb_func` function for every single feature in the input.
                So if you have an input tensor of shape (N, C, H, W) and you set `n_perturbations_per_feature`,
                then each example in the batch will be handled separately and for each example a total of
                `C * H * W` perturbation steps will be performed and in each perturbation step a single feature
                will be repeated `n_perturbations_per_feature` times. This means that the total number of
                perturbation steps will be `C * H * W * n_perturbations_per_feature` for each example in the batch.

                Default: 10
        max_features_processed_per_batch (int, optional): The number of maximum input
                features that are processed together for every example. In case the number of
                features in each example (`C * H * W`) exceeds
                `max_features_processed_per_batch`, they will be sliced
                into batches of `max_features_processed_per_batch` examples and processed
                in a sequential order. However the total effective batch size will still be
                `max_features_processed_per_batch * n_perturbations_per_feature` as in each
                perturbation step, `max_features_processed_per_batch * n_perturbations_per_feature` features
                are processed. If `max_features_processed_per_batch` is None, all
                examples are processed together. `max_features_processed_per_batch` should
                at least be equal `n_perturbations_per_feature` and at most
                `C * H * W * n_perturbations_per_feature`.
        frozen_features (List[torch.Tensor], optional): A list of frozen features that are not perturbed.
                This can be useful for ignoring the input structure features like padding, etc. Default: None
                In case CLS,PAD,SEP tokens are present in the input, they can be frozen by passing the indices
                of feature masks that correspond to these tokens.
        eps (float, optional): Defines the minimum threshold for the attribution scores and the model forward
                variances. If the absolute value of the attribution scores or the model forward variances
                is less than `eps`, it is considered as zero. This is used to compute the non-sensitivity
                metric. Default: 1e-5

        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class.
                Default is False.

        show_progress (bool, optional): Displays the progress of the computation. Default: True
        return_intermediate_results (bool, optional): A boolean flag that indicates whether the intermediate results
                of the metric computation are returned.
                Default is False.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
                with keys as the metric names and values as the corresponding metric outputs.
                Default is False.
    Returns:
        A tuple of tensors:
            monotonicity_corr_batch (Tensor): A tensor of scalar monotonicity_corr scores per
                    input example. The first dimension is equal to the
                    number of examples in the input batch and the second
                    dimension is one.
            non_sensitivity_batch (Tensor): A tensor of scalar non_sensitivity scores per
                    input example. The first dimension is equal to the
                    number of examples in the input batch and the second
                    dimension is one.
            n_features_batch (Tensor): A tensor of scalar values that represent the total number of
                    features that are processed in the input batch. The first dimension is equal to the
                    number of examples in the input batch and the second dimension is one.

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

        >>> # Computes the monotonicity correlation and non-sensitivity scores for saliency maps
        >>> monotonicity_corr, non_sens, n_features = monotonicity_corr_and_non_sens(net, input, attribution, baselines)
    """
    if is_multi_target:
        (
            monotonicity_corr_batch_list,
            non_sens_batch_list,
            n_features_batch,
            perturbed_fwd_diffs_relative_vars_batch_list,
            feature_group_attribution_scores_batch_list,
        ) = _multi_target_monotonicity_corr_and_non_sens(
            forward_func=forward_func,
            inputs=inputs,
            attributions_list=attributions,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            targets_list=target,
            perturb_func=perturb_func,
            n_perturbations_per_feature=n_perturbations_per_feature,
            max_features_processed_per_batch=max_features_processed_per_batch,
            eps=eps,
            frozen_features=frozen_features,
            show_progress=show_progress,
            return_ratio=return_ratio,
        )

        if return_intermediate_results:
            if return_dict:
                return {
                    "monotonicity_corr_score": monotonicity_corr_batch_list,
                    "non_sensitivity_score": non_sens_batch_list,
                    "n_features": n_features_batch,
                    "perturbed_fwd_diffs_relative_vars": perturbed_fwd_diffs_relative_vars_batch_list,
                    "feature_group_attribution_scores": feature_group_attribution_scores_batch_list,
                }
            else:
                return (
                    monotonicity_corr_batch_list,
                    non_sens_batch_list,
                    n_features_batch,
                    perturbed_fwd_diffs_relative_vars_batch_list,
                    feature_group_attribution_scores_batch_list,
                )
        else:
            if return_dict:
                return {
                    "monotonicity_corr_score": monotonicity_corr_batch_list,
                    "non_sensitivity_score": non_sens_batch_list,
                }
            else:
                return monotonicity_corr_batch_list, non_sens_batch_list

    with torch.no_grad():
        # perform argument formattings
        inputs = _format_tensor_into_tuples(inputs)  # type: ignore
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        attributions = _format_tensor_into_tuples(attributions)  # type: ignore
        feature_mask = _format_tensor_into_tuples(feature_mask)  # type: ignore

        # Make sure that inputs and corresponding attributions have matching sizes.
        assert len(inputs) == len(attributions), (
            """The number of tensors in the inputs and
            attributions must match. Found number of tensors in the inputs is: {} and in the
            attributions: {}"""
        ).format(len(inputs), len(attributions))
        if feature_mask is not None:
            for mask, attribution in zip(feature_mask, attributions):
                assert mask.shape == attribution.shape, (
                    """The shape of the feature mask and the attribution
                    must match. Found feature mask shape: {} and attribution shape: {}"""
                ).format(mask.shape, attribution.shape)

        bsz = inputs[0].size(0)
        monotonicity_corr_batch = []
        non_sens_batch = []
        n_features_batch = []
        perturbed_fwd_diffs_relative_vars_batch = []
        feature_group_attribution_scores_batch = []
        for sample_idx in tqdm.tqdm(range(bsz), disable=not show_progress):
            (
                monotonicity_corr,
                non_sens,
                n_features,
                perturbed_fwd_diffs_relative_vars,
                feature_group_attribution_scores,
            ) = eval_monotonicity_corr_and_non_sens_single_sample(
                forward_func=forward_func,
                inputs=tuple(input[sample_idx].unsqueeze(0) for input in inputs),
                attributions=tuple(
                    attr[sample_idx].unsqueeze(0) for attr in attributions
                ),
                feature_mask=(
                    tuple(mask[sample_idx].unsqueeze(0) for mask in feature_mask)
                    if feature_mask is not None
                    else None
                ),
                additional_forward_args=(
                    tuple(
                        (
                            arg[sample_idx].unsqueeze(0)
                            if isinstance(arg, torch.Tensor)
                            else arg
                        )
                        for arg in additional_forward_args
                    )
                    if additional_forward_args is not None
                    else None
                ),
                target=(
                    target[sample_idx]
                    if isinstance(target, (list, torch.Tensor))
                    else target
                ),
                perturb_func=perturb_func,
                n_perturbations_per_feature=n_perturbations_per_feature,
                max_features_processed_per_batch=max_features_processed_per_batch,
                frozen_features=(
                    frozen_features[sample_idx]
                    if frozen_features is not None
                    else frozen_features
                ),
                use_absolute_attributions=use_absolute_attributions,
                eps=eps,
                show_progress=show_progress,
                return_ratio=return_ratio,
            )

            monotonicity_corr_batch.append(monotonicity_corr)
            non_sens_batch.append(non_sens)
            n_features_batch.append(n_features)
            perturbed_fwd_diffs_relative_vars_batch.append(
                perturbed_fwd_diffs_relative_vars
            )
            feature_group_attribution_scores_batch.append(
                feature_group_attribution_scores
            )
        monotonicity_corr_batch = torch.tensor(monotonicity_corr_batch)
        non_sens_batch = torch.tensor(non_sens_batch)
        n_features_batch = torch.tensor(n_features_batch)
        perturbed_fwd_diffs_relative_vars_batch = [
            torch.tensor(x) for x in perturbed_fwd_diffs_relative_vars_batch
        ]
        feature_group_attribution_scores_batch = [
            torch.tensor(x) for x in feature_group_attribution_scores_batch
        ]

        if return_intermediate_results:
            if return_dict:
                return {
                    "monotonicity_corr_score": monotonicity_corr_batch,
                    "non_sensitivity_score": non_sens_batch,
                    "n_features": n_features_batch,
                    "perturbed_fwd_diffs_relative_vars": perturbed_fwd_diffs_relative_vars_batch,
                    "feature_group_attribution_scores": feature_group_attribution_scores_batch,
                }
            else:
                return (
                    monotonicity_corr_batch,
                    non_sens_batch,
                    n_features_batch,
                    perturbed_fwd_diffs_relative_vars_batch,
                    feature_group_attribution_scores_batch,
                )
        else:
            if return_dict:
                return {
                    "monotonicity_corr_score": monotonicity_corr_batch,
                    "non_sensitivity_score": non_sens_batch,
                }
            else:
                return monotonicity_corr_batch, non_sens_batch
