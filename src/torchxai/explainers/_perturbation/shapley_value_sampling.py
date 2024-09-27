from typing import Any, Tuple, Union

from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution, ShapleyValueSampling
from torch import Tensor
from torch.nn import Module

from torchxai.explainers.explainer import Explainer

# class MultiTargetShapleyValueSampling(ShapleyValueSampling):
#     @log_usage()
#     def attribute(
#         self,
#         inputs: TensorOrTupleOfTensorsGeneric,
#         baselines: BaselineType = None,
#         target: TargetType = None,
#         additional_forward_args: Any = None,
#         feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
#         n_samples: int = 25,
#         perturbations_per_eval: int = 1,
#         show_progress: bool = False,
#     ) -> TensorOrTupleOfTensorsGeneric:
#         # Keeps track whether original input is a tuple or not before
#         # converting it into a tuple.
#         is_inputs_tuple = _is_tuple(inputs)
#         inputs, baselines = _format_input_baseline(inputs, baselines)
#         additional_forward_args = _format_additional_forward_args(
#             additional_forward_args
#         )
#         feature_mask = _format_feature_mask(feature_mask, inputs)
#         feature_mask = _shape_feature_mask(feature_mask, inputs)

#         assert (
#             isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
#         ), "Ablations per evaluation must be at least 1."

#         with torch.no_grad():
#             baselines = _tensorize_baseline(inputs, baselines)
#             num_examples = inputs[0].shape[0]

#             total_features = _get_max_feature_index(feature_mask) + 1

#             if show_progress:
#                 attr_progress = progress(
#                     desc=f"{self.get_name()} attribution",
#                     total=self._get_n_evaluations(
#                         total_features, n_samples, perturbations_per_eval
#                     )
#                     + 1,  # add 1 for the initial eval
#                 )
#                 attr_progress.update(0)

#             initial_eval = self._strict_run_forward(
#                 self.forward_func, baselines, target, additional_forward_args
#             )

#             if show_progress:
#                 attr_progress.update()

#             agg_output_mode = _find_output_mode_and_verify(
#                 initial_eval,
#                 num_examples,
#                 perturbations_per_eval,
#                 feature_mask,
#                 allow_multi_outputs=True,
#             )

#             # Initialize attribution totals and counts
#             output_shape = initial_eval.shape

#             # attr shape (*output_shape, *input_feature_shape)
#             total_attrib = [
#                 torch.zeros(
#                     output_shape + input.shape[1:],
#                     dtype=torch.float,
#                     device=inputs[0].device,
#                 )
#                 for input in inputs
#             ]

#             total_attrib = [
#                 # attribute w.r.t each output element
#                 torch.zeros(
#                     (n_outputs[0] * n_outputs[1],) + input.shape[1:],
#                     dtype=attrib_type,
#                     device=input.device,
#                 )
#                 for input in inputs
#             ]

#             iter_count = 0
#             # Iterate for number of samples, generate a permutation of the features
#             # and evalute the incremental increase for each feature.
#             for feature_permutation in self.permutation_generator(
#                 total_features, n_samples
#             ):
#                 iter_count += 1
#                 prev_results = initial_eval
#                 for (
#                     current_inputs,
#                     current_add_args,
#                     current_target,
#                     current_masks,
#                 ) in self._perturbation_generator(
#                     inputs,
#                     additional_forward_args,
#                     target,
#                     baselines,
#                     feature_mask,
#                     feature_permutation,
#                     perturbations_per_eval,
#                 ):
#                     if sum(torch.sum(mask).item() for mask in current_masks) == 0:
#                         warnings.warn(
#                             "Feature mask is missing some integers between 0 and "
#                             "num_features, for optimal performance, make sure each"
#                             " consecutive integer corresponds to a feature."
#                         )
#                     # modified_eval dimensions: 1D tensor with length
#                     # equal to #num_examples * #features in batch
#                     modified_eval = self._strict_run_forward(
#                         self.forward_func,
#                         current_inputs,
#                         current_target,
#                         current_add_args,
#                     )
#                     if show_progress:
#                         attr_progress.update()

#                     if agg_output_mode:
#                         eval_diff = modified_eval - prev_results
#                         prev_results = modified_eval
#                     else:
#                         # when perturb_per_eval > 1, every num_examples stands for
#                         # one perturb. Since the perturbs are from a consecutive
#                         # perumuation, each diff of a perturb is its eval minus
#                         # the eval of the previous perturb
#                         all_eval = torch.cat((prev_results, modified_eval), dim=0)
#                         eval_diff = all_eval[num_examples:] - all_eval[:-num_examples]
#                         prev_results = all_eval[-num_examples:]

#                     for j in range(len(total_attrib)):
#                         # format eval_diff to shape
#                         # (n_perturb, *output_shape, 1,.. 1)
#                         # where n_perturb may not be perturb_per_eval
#                         # Append n_input_feature dim of 1 to make the tensor
#                         # have the same dim as the mask tensor.
#                         formatted_eval_diff = eval_diff.reshape(
#                             (-1,) + output_shape + (len(inputs[j].shape) - 1) * (1,)
#                         )

#                         # mask in shape (n_perturb, *mask_shape_broadcastable_to_input)
#                         # reshape to
#                         # (
#                         #     n_perturb,
#                         #     *broadcastable_to_output_shape
#                         #     *broadcastable_to_input_feature_shape
#                         # )
#                         cur_mask = current_masks[j]
#                         cur_mask = cur_mask.reshape(
#                             cur_mask.shape[:2]
#                             + (len(output_shape) - 1) * (1,)
#                             + cur_mask.shape[2:]
#                         )

#                         # aggregate n_perturb
#                         cur_attr = (formatted_eval_diff * cur_mask.float()).sum(dim=0)

#                         # (*output_shape, *input_feature_shape)
#                         total_attrib[j] += cur_attr

#             if show_progress:
#                 attr_progress.close()

#             # Divide total attributions by number of random permutations and return
#             # formatted attributions.
#             attrib = tuple(
#                 tensor_attrib_total / iter_count for tensor_attrib_total in total_attrib
#             )
#             formatted_attr = _format_output(is_inputs_tuple, attrib)
#         return formatted_attr


class ShapleyValueSamplingExplainer(Explainer):
    """
    A Explainer class for handling Shapley Values using the Captum library.

    Args:
        model (torch.nn.Module): The model whose output is to be explained.
        n_samples (int, optional): The number of samples to use for Shapley Values. Default is 200.
        perturbations_per_eval (int, optional): The number of perturbations evaluated per batch. Default is 50.

    Attributes:
        n_samples (int): The number of samples to be used for generating Shapley Values attributions.
        perturbations_per_eval (int): The number of perturbations evaluated per batch.
    """

    REQUIRES_FEATURE_MASK = True

    def __init__(
        self,
        model: Module,
        n_samples: int = 200,
        perturbations_per_eval: int = 50,
    ) -> None:
        """
        Initialize the ShapleyValueSamplingExplainer with the model and number of samples.
        """
        super().__init__(model)
        self.n_samples = n_samples
        self.perturbations_per_eval = perturbations_per_eval

    def _init_explanation_fn(self) -> Attribution:
        """
        Initializes the explanation function.

        Returns:
            Attribution: The initialized explanation function.
        """
        if self._is_multi_target:
            raise NotImplementedError(
                "Multi-target Shapley Value Sampling is not supported."
            )
        return ShapleyValueSampling(self._model)

    def explain(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType,
        baselines: BaselineType = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        additional_forward_args: Any = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Compute the Shapley Values attributions for the given inputs.

        Args:
            inputs (TensorOrTupleOfTensorsGeneric): The input tensor(s) for which attributions are computed.
            target (TargetType): The target(s) for computing attributions.
            baselines (BaselineType, optional): Baselines for computing attributions. Default is None.
            feature_mask (Union[None, Tensor, Tuple[Tensor, ...]], optional): Masks representing feature groups. Default is None.
            additional_forward_args (Any, optional): Additional arguments to forward to the model. Default is None.

        Returns:
            TensorOrTupleOfTensorsGeneric: The computed attributions.
        """
        # Compute the attributions using Shapley Values
        attributions = self._explanation_fn.attribute(
            inputs=inputs,
            target=target,
            baselines=baselines,
            feature_mask=feature_mask,
            additional_forward_args=additional_forward_args,
            n_samples=self.n_samples,
            perturbations_per_eval=self.perturbations_per_eval,
            show_progress=True,
        )

        return attributions
