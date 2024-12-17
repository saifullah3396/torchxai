from typing import List, Optional, Tuple, Union

import torch

from torchxai.metrics._utils.common import (
    _construct_default_feature_mask,
    _reduce_tensor_with_indices_non_deterministic,
    _tuple_tensors_to_tensors,
    _validate_feature_mask,
)


def _complexity_sundararajan(
    attributions: Union[Tuple[torch.Tensor, ...], List[Tuple[torch.Tensor, ...]]],
    eps: float = 1e-5,
    normalize_attribution: bool = True,
) -> torch.Tensor:
    with torch.no_grad():
        if not isinstance(attributions, tuple):
            attributions = (attributions,)

        # get batch size
        bsz = attributions[0].shape[0]

        # flatten feature tuple tensors into a single tensor of shape [batch_size, num_features]
        attributions, _ = _tuple_tensors_to_tensors(attributions)

        # flatten the feature dims into a single dim
        attributions = attributions.view(bsz, -1).float().abs()

        # normalize the attributions sample-wise in the batch if required
        if normalize_attribution:
            attributions = attributions / torch.sum(attributions, dim=1, keepdim=True)

        # compute batch-wise complexity of the attribution map
        complexity_sundararajan_score = torch.sum(
            attributions > eps, dim=1
        ) / attributions.size(1)
        return complexity_sundararajan_score


def _complexity_sundararajan_feature_grouped(
    attributions: Union[Tuple[torch.Tensor, ...], List[Tuple[torch.Tensor, ...]]],
    feature_mask: Optional[Tuple[torch.Tensor, ...]],
    eps: float = 1e-5,
    normalize_attribution: bool = True,
) -> torch.Tensor:

    with torch.no_grad():

        def complexity_sundararajan_feature_grouped_single_sample(
            attributions_single_sample, feature_mask_single_sample
        ):
            if not isinstance(attributions_single_sample, tuple):
                attributions_single_sample = (attributions_single_sample,)

            # get batch size
            bsz = attributions_single_sample[0].shape[0]
            assert bsz == 1, "Batch size must be 1 for feature grouped complexity"

            # flatten all feature masks in the input
            if feature_mask_single_sample is not None:
                feature_mask_flattened, _ = _tuple_tensors_to_tensors(
                    feature_mask_single_sample
                )
            else:
                feature_mask_single_sample = _construct_default_feature_mask(
                    attributions_single_sample
                )
                feature_mask_flattened, _ = _tuple_tensors_to_tensors(
                    feature_mask_single_sample
                )

            # flatten all attributions_single_sample in the input, this must be done after the feature masks are flattened as
            # feature masks may depened on attribution
            attributions_single_sample, _ = _tuple_tensors_to_tensors(
                attributions_single_sample
            )

            # validate feature masks are increasing non-negative
            _validate_feature_mask(feature_mask_flattened)

            # gather attribution scores of feature groups
            # this can be useful for efficiently summing up attributions of feature groups
            # this is why we need a single batch size as gathered attributes and number of features for each
            # sample can be different
            reduced_attributions, n_features = (
                _reduce_tensor_with_indices_non_deterministic(
                    attributions_single_sample[0],
                    indices=feature_mask_flattened[0].flatten(),
                )
            )
            reduced_attributions = reduced_attributions.abs()

            # normalize the attributions to get fractional contribution
            if normalize_attribution:
                reduced_attributions = reduced_attributions / torch.sum(
                    reduced_attributions
                )
            # compute batch-wise effective complexity of the attribution map
            return torch.sum(reduced_attributions > eps) / n_features

        if not isinstance(attributions, tuple):
            attributions = (attributions,)
        bsz = attributions[0].size(0)
        complexity_sundararajan_score_batch = []
        for sample_idx in range(bsz):
            complexity_sundararajan_score = (
                complexity_sundararajan_feature_grouped_single_sample(
                    attributions_single_sample=tuple(
                        attr[sample_idx].unsqueeze(0) for attr in attributions
                    ),
                    feature_mask_single_sample=(
                        tuple(mask[sample_idx].unsqueeze(0) for mask in feature_mask)
                        if feature_mask is not None
                        else None
                    ),
                )
            )
            complexity_sundararajan_score_batch.append(complexity_sundararajan_score)
        return torch.tensor(complexity_sundararajan_score_batch)


def complexity_sundararajan(
    attributions: Union[Tuple[torch.Tensor, ...], List[Tuple[torch.Tensor, ...]]],
    eps: float = 1e-5,
    normalize_attribution: bool = True,
    is_multi_target: bool = False,
    return_dict: bool = False,
) -> torch.Tensor:
    """
    Implementation of Complexity metric by Sundararajan at el., 2017. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library.

    Complexity measures how many attributions in absolute values are exceeding a certain threshold (eps)
    where a value above the specified threshold implies that the features are important and under indicates it is not.
    Complexity requires the attributions to be normalized to return reasonable outputs since the original
    attributions may have different scales and effective complexity is sensitive to the scale of the attributions.

    References:
        1) Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic Attribution for Deep Networks.
        In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML’17,
        pages 3319–3328. JMLR.org, 2017.
        1) An-phi Nguyen and María Rodríguez Martínez.: "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).

    Args:
        attributions (Tuple[Tensor,...]): A tuple of tensors representing attributions of separate inputs. Each
            tensor in the tuple has shape (batch_size, num_features).
        eps (float): The threshold value for attributions to be considered important.
        normalize_attribution (bool): If True, the attributions are normalized to sum to 1.
        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class.
                Default is False.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
                with keys as the metric names and values as the corresponding metric outputs.
                Default is False.
    Returns:
        Tensor: A tensor of scalar effective complexity per
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

        >>> # Computes the monotonicity correlation and non-sensitivity scores for saliency maps
        >>> effective_complexity_scores = effective_complexity(attribution)
    """
    is_attributions_list = isinstance(attributions, list)
    if is_multi_target:
        assert (
            is_attributions_list
        ), "attributions must be a list of tensors or list of tuples of tensors"
    if not is_attributions_list:
        attributions = [attributions]
    score = [
        _complexity_sundararajan(
            attributions=attribution,
            eps=eps,
            normalize_attribution=normalize_attribution,
        )
        for attribution in attributions
    ]
    if not is_attributions_list:
        score = score[0]
    if return_dict:
        return {"complexity_sundararajan_score": score}
    return score


def complexity_sundararajan_feature_grouped(
    attributions: Union[Tuple[torch.Tensor, ...], List[Tuple[torch.Tensor, ...]]],
    feature_mask: Optional[Tuple[torch.Tensor, ...]] = None,
    eps: float = 1e-5,
    normalize_attribution: bool = True,
    is_multi_target: bool = False,
    return_dict: bool = False,
) -> torch.Tensor:
    """
    Implementation of Complexity metric by Sundararajan at el., 2017. This implementation
    reuses the batch-computation ideas from captum and therefore it is fully compatible with the Captum library.
    In addition, the implementation takes some ideas about the implementation of the metric from the python
    Quantus library. In this particular implementation, the attributions are grouped into feature groups and the
    complexity is computed based on the entropy of the fractional contribution of feature groups to the total
    magnitude of the attribution.


    Complexity measures how many attributions in absolute values are exceeding a certain threshold (eps)
    where a value above the specified threshold implies that the features are important and under indicates it is not.
    Complexity requires the attributions to be normalized to return reasonable outputs since the original
    attributions may have different scales and effective complexity is sensitive to the scale of the attributions.

    References:
        1) Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic Attribution for Deep Networks.
        In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML’17,
        pages 3319–3328. JMLR.org, 2017.
        1) An-phi Nguyen and María Rodríguez Martínez.: "On quantitative aspects of model
        interpretability." arXiv preprint arXiv:2007.07584 (2020).

    Args:
        attributions (Tuple[Tensor,...]): A tuple of tensors representing attributions of separate inputs. Each
            tensor in the tuple has shape (batch_size, num_features).
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
        eps (float): The threshold value for attributions to be considered important.
        normalize_attribution (bool): If True, the attributions are normalized to sum to 1.
        is_multi_target (bool, optional): A boolean flag that indicates whether the metric computation is for
                multi-target explanations. if set to true, the targets are required to be a list of integers
                each corresponding to a required target class in the output. The corresponding metric outputs
                are then returned as a list of metric outputs corresponding to each target class.
                Default is False.
        return_dict (bool, optional): A boolean flag that indicates whether the metric outputs are returned as a dictionary
                with keys as the metric names and values as the corresponding metric outputs.
                Default is False.
    Returns:
        Tensor: A tensor of scalar effective complexity per
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

        >>> # Computes the monotonicity correlation and non-sensitivity scores for saliency maps
        >>> effective_complexity_scores = effective_complexity(attribution)
    """

    is_attributions_list = isinstance(attributions, list)
    if is_multi_target:
        assert (
            is_attributions_list
        ), "attributions must be a list of tensors or list of tuples of tensors"
    if not is_attributions_list:
        attributions = [attributions]

    score = [
        _complexity_sundararajan_feature_grouped(
            attributions=attribution,
            feature_mask=feature_mask,
            eps=eps,
            normalize_attribution=normalize_attribution,
        )
        for attribution in attributions
    ]
    if not is_attributions_list:
        score = score[0]
    if return_dict:
        return {"complexity_sundararajan_feature_grouped_score": score}
    return score
