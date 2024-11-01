#!/usr/bin/env python3

from typing import Any, Callable, Optional

from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from torch import Tensor

from torchxai.metrics.faithfulness.multi_target.infidelity import (
    _multi_target_infidelity,
)


def infidelity(
    forward_func: Callable,
    perturb_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    n_perturb_samples: int = 10,
    max_examples_per_batch: Optional[int] = None,
    normalize: bool = False,
    is_multi_target: bool = False,
    return_dict: bool = False,
) -> Tensor:
    r"""
    A wrapper around the Captum library's infidelity metric. The metric returns a list of infidelity
    scores if is_multi_target is True using the `torchxai.metrics.faithfulness.multi_target._multi_target_infidelity`,
    otherwise it returns a single infidelity score using the captum implementation `captum.metrics.infidelity`.

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

        perturb_func (Callable):
                The perturbation function of model inputs. This function takes
                model inputs and optionally baselines as input arguments and returns
                either a tuple of perturbations and perturbed inputs or just
                perturbed inputs. For example:

                >>> def my_perturb_func(inputs):
                >>>   <MY-LOGIC-HERE>
                >>>   return perturbations, perturbed_inputs

                If we want to only return perturbed inputs and compute
                perturbations internally then we can wrap perturb_func with
                `infidelity_perturb_func_decorator` decorator such as:

                >>> from captum.metrics import infidelity_perturb_func_decorator

                >>> @infidelity_perturb_func_decorator(<multipy_by_inputs flag>)
                >>> def my_perturb_func(inputs):
                >>>   <MY-LOGIC-HERE>
                >>>   return perturbed_inputs

                `infidelity_perturb_func_decorator` needs to be used with
                `multipy_by_inputs` flag set to False in case infidelity
                score is being computed for attribution maps that are local aka
                that do not factor in inputs in the final attribution score.
                Such attribution algorithms include Saliency, GradCam, Guided Backprop,
                or Integrated Gradients and DeepLift attribution scores that are already
                computed with `multipy_by_inputs=False` flag.

                If there are more than one inputs passed to infidelity function those
                will be passed to `perturb_func` as tuples in the same order as they
                are passed to infidelity function.

                If inputs
                 - is a single tensor, the function needs to return a tuple
                   of perturbations and perturbed input such as:
                   perturb, perturbed_input and only perturbed_input in case
                   `infidelity_perturb_func_decorator` is used.
                 - is a tuple of tensors, corresponding perturbations and perturbed
                   inputs must be computed and returned as tuples in the
                   following format:

                   (perturb1, perturb2, ... perturbN), (perturbed_input1,
                   perturbed_input2, ... perturbed_inputN)

                   Similar to previous case here as well we need to return only
                   perturbed inputs in case `infidelity_perturb_func_decorator`
                   decorates out `perturb_func`.

                It is important to note that for performance reasons `perturb_func`
                isn't called for each example individually but on a batch of
                input examples that are repeated `max_examples_per_batch / batch_size`
                times within the batch.

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
        >>> # Computes infidelity score for saliency maps
        >>> infid = infidelity(net, perturb_fn, input, attribution)
    """
    if is_multi_target:
        infidelity_score = _multi_target_infidelity(
            forward_func=forward_func,
            perturb_func=perturb_func,
            inputs=inputs,
            attributions_list=attributions,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            targets_list=target,
            n_perturb_samples=n_perturb_samples,
            max_examples_per_batch=max_examples_per_batch,
            normalize=normalize,
        )
        if return_dict:
            return {"infidelity_score": infidelity_score}
        return infidelity_score

    from captum.metrics import infidelity

    infidelity_score = infidelity(
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

    if return_dict:
        return {"infidelity_score": infidelity_score}
    return infidelity_score
