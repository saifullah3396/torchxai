from typing import Any, Callable, List, Tuple, Union, cast

import torch
from captum._utils.common import (
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    _run_forward,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor
from torchxai.metrics.axiomatic.multi_target.completeness import (
    _multi_target_completeness,
)


def _completeness(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType,
    additional_forward_args: Any = None,
    target: TargetType = None,
) -> Union[Tensor, List[Tensor]]:
    with torch.no_grad():
        inputs = _format_tensor_into_tuples(inputs)  # type: ignore
        if baselines is None:
            baselines = tuple(torch.zeros_like(inp) for inp in inputs)
        else:
            baselines = _format_baseline(baselines, cast(Tuple[Tensor, ...], inputs))
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        attributions = _format_tensor_into_tuples(attributions)  # type: ignore

        # Make sure that inputs and corresponding attributions have number of tuples.
        assert len(inputs) == len(attributions), (
            """The number of tensors in the inputs and
          attributions must match. Found number of tensors in the inputs is: {} and in the
          attributions: {}"""
        ).format(len(inputs), len(attributions))

        # for this implementation the shapes of the inputs and attributions are not necessarily needed to be matched
        # for example the inputs can be of shape (batch_size, seq_length, n_features) and the attributions can be of shape
        # (batch_size, seq_length) where the n_features dim is summed for seq_length.

        # get the batch size
        bsz = inputs[0].size(0)

        # compute the forward pass on inputs
        inputs_fwd = _run_forward(forward_func, inputs, target, additional_forward_args)

        # compute the forward pass on baselines
        baselines_fwd = _run_forward(
            forward_func,
            baselines,
            target,
            additional_forward_args,
        )

        # compute the difference between the forward pass on inputs and baselines
        fwd_diffs = inputs_fwd - baselines_fwd

        # compute the sum of attributions
        attributions_sum = sum(tuple(x.view(bsz, -1).sum(dim=-1) for x in attributions))

        # compute the absolute difference between the sum of attributions and the forward pass difference
        # this is the completeness score, the lower the score the better the completeness
        return torch.abs(attributions_sum - fwd_diffs)


def completeness(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: Union[
        List[TensorOrTupleOfTensorsGeneric], TensorOrTupleOfTensorsGeneric
    ],
    baselines: BaselineType,
    additional_forward_args: Any = None,
    target: TargetType = None,
    is_multi_target: bool = False,
    return_dict: bool = False,
):
    """
    Implementation of Completeness test by Sundararajan et al., 2017, also referred
    to as Summation to Delta by Shrikumar et al., 2017 and Conservation by
    Montavon et al., 2018. This implementation reuses the batch-computation ideas from captum and therefore it is fully
    compatible with the Captum library. In addition, the implementation takes some ideas about the implementation
    of the metric from the python Quantus library.

    Attribution completeness asks that the total attribution is proportional to the explainable
    evidence at the output/ or some function of the model output. Or, that the attributions
    add up to the difference between the model output F at the input x and the baseline b. This is essentially
    the same as `convergence_delta` returned from captum however this works independently of the attribution method
    and the availability of `convergence_delta` calculation in Captum.

    References:
        1) Completeness - Mukund Sundararajan et al.: "Axiomatic attribution for deep networks."
        International Conference on Machine Learning. PMLR, 2017.
        2) Summation to delta - Avanti Shrikumar et al.: "Learning important
        features through propagating activation differences." International Conference on Machine Learning. PMLR, 2017.
        3) Conservation - Grégoire Montavon et al.: "Methods for interpreting
        and understanding deep neural networks." Digital Signal Processing 73 (2018): 1-15.

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

                Attributions have the same shape and dimensionality as the inputs.
                If inputs is a single tensor then the attributions is a single
                tensor as well. If inputs is provided as a tuple of tensors
                then attributions will be tuples of tensors as well.

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
        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class.
                Default is False.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
                with keys as the metric names and values as the corresponding metric outputs.
                Default is False.
    Returns:
        Tensor: A tensor of scalar completeness scores per
                input example. The first dimension is equal to the
                number of examples in the input batch and the second
                dimension is one.

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

        >>> # Computes completeness score for saliency maps
        >>> completeness = completeness(net, input, attribution, baselines)
    """
    metric_func = _multi_target_completeness if is_multi_target else _completeness
    completeness_score = metric_func(
        forward_func=forward_func,
        inputs=inputs,
        **(
            dict(attributions_list=attributions)
            if is_multi_target
            else dict(attributions=attributions)
        ),
        baselines=baselines,
        additional_forward_args=additional_forward_args,
        **dict(targets_list=target) if is_multi_target else dict(target=target),
    )
    if return_dict:
        return {"completeness_score": completeness_score}
    return completeness_score
