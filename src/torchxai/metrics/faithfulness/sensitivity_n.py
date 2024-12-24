import itertools
from typing import Any, Callable, List, Optional, Union

import torch
from captum._utils.common import _format_tensor_into_tuples
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor
from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _validate_feature_mask,
)
from torchxai.metrics._utils.perturbation import _generate_random_perturbation_masks
from torchxai.metrics.faithfulness.infidelity import _infidelity
from torchxai.metrics.faithfulness.multi_target.infidelity import (
    _multi_target_infidelity,
)


def sensitivity_n(
    n_features_perturbed: Union[int, float],
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: Union[
        List[TensorOrTupleOfTensorsGeneric], TensorOrTupleOfTensorsGeneric
    ],
    baselines: BaselineType,
    feature_mask: TensorOrTupleOfTensorsGeneric = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    n_perturb_samples: int = 10,
    max_examples_per_batch: Optional[int] = None,
    frozen_features: Optional[List[torch.Tensor]] = None,
    normalize: bool = False,
    is_multi_target: bool = False,
    return_dict: bool = False,
) -> Tensor:
    r"""
    A wrapper around the Captum library's infidelity metric that computes senstivity_n.
    The metric returns a list of senstivity_n scores if is_multi_target is True using the
    `torchxai.metrics.faithfulness.multi_target._multi_target_infidelity`,
    otherwise it returns a single sensitivity_n score using the captum implementation `captum.metrics.infidelity`.

    Sensitivity-n takes the same implementation as infidelity but defines a fixed perturbation function
    that perturbs n features for each sample at a time.

    Explanation infidelity represents the expected mean-squared error
    between the explanation multiplied by a meaningful input perturbation
    and the differences between the predictor function at its input
    and perturbed input.
    More details about the measure can be found in the following paper:
    https://arxiv.org/abs/1901.09392

    It is derived from the completeness property of well-known attribution
    algorithms and is a computationally more efficient and generalized
    notion of Sensitivy-n. The latter measures correlations between the sum
    of the attributions and the differences of the predictor function at
    its input and fixed baseline. More details about the Sensitivity-n can
    be found here:
    https://arxiv.org/abs/1711.06104

    The users can perturb the inputs any desired way by providing any
    perturbation function that takes the inputs (and optionally baselines)
    and returns perturbed inputs or perturbed inputs and corresponding
    perturbations.

    This specific implementation is primarily tested for attribution-based
    explanation methods but the idea can be expanded to use for non
    attribution-based interpretability methods as well.

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

        baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                Baselines define reference values which sometimes represent ablated
                values and are used to compare with the actual inputs to compute
                importance scores in attribution algorithms. For sensitivity-n baselines are required
                to compute the perturbations. Baselines can be provided as:
                They can be represented
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
        n_features_perturbed (int or float): Number of features to perturb for each sample at a time.
                This n corresponds to the sensitivity-n value. The perturbation function will perturb n
                features for each sample n_perturb_samples times to compute the sensitivity-n for each sample.
                Alternatively, if n_features_perturbed is a float, it will be interpreted as a percentage of the total
                number of features in the input. For example, if n_features_perturbed=0.1, 10% of the total number of features
                will be perturbed for each sample at a time.
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
        normalize (bool, optional): Normalize the dot product of the input
                perturbation and the attribution so the infidelity value is invariant
                to constant scaling of the attribution values. The normalization factor
                beta is defined as the ratio of two mean values:

                .. math::
                    \beta = \frac{
                        \mathbb{E}_{I \sim \mu_I} [ I^T \Phi(f, x) (f(x) - f(x - I)) ]
                    }{
                        \mathbb{E}_{I \sim \mu_I} [ (I^T \Phi(f, x))^2 ]
                    }

                Please refer the original paper for the meaning of the symbols. Same
                normalization can be found in the paper's official implementation
                https://github.com/chihkuanyeh/saliency_evaluation

                Default: False
        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class. For multi-target
                infidelity, captum implementation is extened in _multi_target_infidelity function.
                Default is False.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
                with keys as the metric names and values as the corresponding metric outputs.
                Default is False.
    Returns:

        infidelities (Tensor): A tensor of scalar infidelity scores per
                input example. The first dimension is equal to the
                number of examples in the input batch and the second
                dimension is one.

    Examples::
        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> saliency = Saliency(net)
        >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> # Computes saliency maps for class 3.
        >>> attribution = saliency.attribute(input, target=3)
        >>> # define a perturbation function for the input
        >>> def perturb_fn(inputs):
        >>>    noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float()
        >>>    return noise, inputs - noise
        >>> # Computes sensitivity_n score for saliency maps
        >>> infid = sensitivity_n(net, n_features_perturbed=1, baselines=0, input, attribution)
    """
    if isinstance(n_features_perturbed, float):
        assert (
            0 < n_features_perturbed <= 1
        ), "If n_features_perturbed is a float, it should be in the range (0, 1]"
    elif isinstance(n_features_perturbed, int):
        assert (
            n_features_perturbed > 0
        ), "n_features_perturbed should be greater than 0. This defines the N in sensitivity-N"
    else:
        raise ValueError(
            "n_features_perturbed should be either an int or a float between (0, 1]"
        )

    feature_mask = _format_tensor_into_tuples(feature_mask)  # type: ignore
    if feature_mask is None:
        feature_mask = _construct_default_feature_mask(attributions)

    # assert that all elements in the feature_mask are unique and non-negative increasing
    _validate_feature_mask(feature_mask)

    batch_size = inputs[0].shape[0] if isinstance(inputs, tuple) else inputs.shape[0]

    def sensitivity_perturb_function(inputs, baselines):
        # each input is repeated max_examples_per_batch / batch_size times
        # so sample 1 is repeated max_examples_per_batch / batch_size times,
        # then sample 2 is repeated max_examples_per_batch / batch_size times, etc.
        # get input shape and number of samples
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if not isinstance(baselines, tuple):
            baselines = (baselines,)

        baselines = (
            tuple(
                torch.ones_like(input) * baseline
                for input, baseline in zip(inputs, itertools.cycle(baselines))
            )
            if isinstance(baselines[0], int)
            else baselines
        )
        n_features = max(tuple(x.max().item() for x in feature_mask)) + 1
        percent_features_perturbed = (
            n_features_perturbed / n_features
            if n_features_perturbed >= 1
            else n_features_perturbed
        )
        total_repeated_inputs = inputs[0].shape[0]
        n_perturbations = total_repeated_inputs // batch_size
        perturbation_masks = _generate_random_perturbation_masks(
            n_perturbations,
            feature_mask,
            percent_features_perturbed=percent_features_perturbed,
            frozen_features=frozen_features,
        )
        perturbation_masks = tuple(
            mask.view_as(input) for mask, input in zip(perturbation_masks, inputs)
        )
        assert perturbation_masks[0].dtype == torch.bool
        return perturbation_masks, tuple(
            input * ~mask + mask * baseline
            for input, mask, baseline in zip(inputs, perturbation_masks, baselines)
        )

    metric_func = _multi_target_infidelity if is_multi_target else _infidelity
    score = metric_func(
        forward_func=forward_func,
        perturb_func=sensitivity_perturb_function,
        inputs=inputs,
        **(
            dict(attributions_list=attributions)
            if is_multi_target
            else dict(attributions=attributions)
        ),
        baselines=baselines,
        additional_forward_args=additional_forward_args,
        **dict(targets_list=target) if is_multi_target else dict(target=target),
        feature_mask=feature_mask,
        frozen_features=frozen_features,
        n_perturb_samples=n_perturb_samples,
        max_examples_per_batch=max_examples_per_batch,
        normalize=normalize,
    )

    if return_dict:
        return {"sensitivity_n_score": score}
    return score
