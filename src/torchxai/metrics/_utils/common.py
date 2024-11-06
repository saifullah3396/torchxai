from typing import Tuple

import torch


def _validate_feature_mask(tensor: Tuple[torch.Tensor, ...]) -> None:
    """
    Validates the feature mask tensor. The feature mask must contain non-negative integers that are strictly increasing
    for example, [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4] is a valid feature mask tensor.
    Args:
        tensor (Tuple[torch.Tensor, ...]): The feature mask tensor.
    Returns:
        bool: True if the feature mask tensor is valid, False otherwise.
    Raises:
        AssertionError: If the tensor is not of tuple type.
        AssertionError: If the tensor values are not non-negative integers.
        AssertionError: If the tensor values are not strictly increasing with a max step size of one.
    """

    if not isinstance(tensor, tuple):
        tensor = (tensor,)

    bsz = tensor[0].size(0)
    flat_tensor = torch.cat([x.contiguous().view(bsz, -1) for x in tensor], dim=-1)

    for t in flat_tensor:
        # Check if all elements are non-negative integers
        # Assert that the tensor is of an integer type
        assert t.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }, "Tensor must be of an integer dtype (torch.int8, torch.int16, torch.int32, or torch.int64)."
        assert torch.all(t >= 0), "tensor values must be non-negative integers"

        # get the unique values in the tensor
        unique_vals = torch.unique(t)

        diff = unique_vals[1:] - unique_vals[:-1]
        assert torch.all(diff >= 0) and torch.all(
            diff <= 1
        ), "tensor values must be strictly increasing with a step of one"


def _construct_default_feature_mask(
    attributions: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    """
    Constructs feature masks from the attributions.
    Args:
        attribution (Tuple[torch.Tensor, ...]): The attributions which is a torch.Tensor or a tuple of torch.Tensors.
    Returns:
        Tuple[torch.Tensor, ...]: The feature masks corresponding to each feature in the input tensor.
    """
    if isinstance(attributions, torch.Tensor):
        attributions = (attributions,)

    last_n_features = 0
    feature_mask = []
    bsz = attributions[0].size(0)
    for attribution in attributions:
        # take each feature in attribution as a feature group and assign an increasing integer index to it
        n_features_in_attribution = attribution.view(bsz, -1).size(-1)
        feature_mask.append(
            (
                torch.arange(n_features_in_attribution, device=attribution.device)
                + last_n_features
            )
            .repeat(bsz, *([1] * len(attribution.shape[1:])))
            .view_as(attribution)
        )
        last_n_features += n_features_in_attribution

    feature_mask = tuple(feature_mask)

    _validate_feature_mask(feature_mask)
    return feature_mask


def _format_tensor_tuple_feature_dim(
    tuple_tensors: Tuple[torch.Tensor],
) -> Tuple[torch.Tensor]:
    return tuple(_format_tensor_feature_dim(x) for x in tuple_tensors)


def _format_tensor_feature_dim(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 1:
        # if there is no feature dim add it to the tensor, this could be a tensor with a single feature
        # this will convert a tensor from shape (batch_size) to (batch_size, 1)
        return tensor.unsqueeze(-1)
    return tensor


def _tuple_tensors_to_tensors(
    tuple_tensors: Tuple[torch.Tensor],
) -> torch.Tensor:
    # first dimension is always batch size
    # if there is no feature dim add it to the tensor, this could be a tensor with a single feature
    if not isinstance(tuple_tensors, tuple):
        return tuple_tensors, tuple_tensors.shape
    if len(tuple_tensors[0].shape) == 1:
        tuple_tensors = tuple(x.unsqueeze(-1) for x in tuple_tensors)
    return (
        torch.cat(tuple(x.reshape(x.shape[0], -1) for x in tuple_tensors), dim=1),
        tuple(x.shape[1:] for x in tuple_tensors),
    )


def _split_tensors_to_tuple_tensors(
    tensor: Tuple[torch.Tensor], shapes, dim=1
) -> torch.Tensor:
    import numpy as np

    assert len(tensor.shape) == 2
    tensor_tuple = ()
    last_size = 0
    for shape in shapes:
        size = np.prod(tuple(shape))
        bsz = tensor.shape[0]
        tensor_tuple += (
            tensor[:, last_size : last_size + size].view(bsz, *tuple(shape)),
        )
        last_size += size
    return tensor_tuple


def _reduce_tensor_with_indices_non_deterministic(
    source: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    source = source.flatten()
    reduced_attributions = torch.zeros(indices.max() + 1, dtype=source.dtype)
    reduced_attributions.index_add_(
        0,
        indices.cpu(),
        source.cpu(),
    )
    n_features = (indices.max() + 1).item()
    return reduced_attributions, n_features


def _reduce_tensor_with_indices(
    source: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    source = source.flatten()
    reduced_attributions = torch.zeros(
        indices.max() + 1, dtype=source.dtype, device=source.device
    )
    for i in range(reduced_attributions.shape[0]):
        reduced_attributions[i] = source[indices == i].sum()
    n_features = (indices.max() + 1).item()
    return reduced_attributions, n_features


def _draw_perturbated_inputs(perturbed_inputs):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(figsize=(20, 10))
    axes.matshow(
        perturbed_inputs[:, :100, 0].cpu().numpy() == 0,
        cmap="viridis",
    )
    plt.xticks(range(1, 54))
    plt.yticks(range(1, 24))
    plt.tick_params(axis="x", bottom=False)
    plt.grid(c="indigo", ls=":", lw="0.4")
    plt.show()


def _draw_perturbated_inputs_with_splits(perturbed_inputs, inputs_shape):
    splitted_perturbed_inputs = _split_tensors_to_tuple_tensors(
        perturbed_inputs, inputs_shape, dim=1
    )

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Define the number of columns in the subplot grid
    ncols = len(
        splitted_perturbed_inputs
    )  # Number of columns equals number of split inputs
    nrows = 2  # Two rows: one for the full span plot and one for split inputs

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(
        nrows,
        ncols,
        width_ratios=[
            x.shape[1] / perturbed_inputs.shape[1] for x in splitted_perturbed_inputs
        ],
    )

    # Create a subplot that spans across all columns
    ax_full_span = fig.add_subplot(gs[0, :])  # First row, all columns
    ax_full_span.matshow(
        perturbed_inputs[:, :, 0].cpu().numpy() == 0,
        cmap="viridis",
    )
    ax_full_span.set_axis_off()
    ax_full_span.set_axis_off()

    # Create subplots for each split input
    for idx, splitted_perturbed_input in enumerate(splitted_perturbed_inputs):
        ax = fig.add_subplot(gs[1, idx])  # Second row, each column
        ax.matshow(
            splitted_perturbed_input[:, :, 0].cpu().numpy() == 0,
            cmap="viridis",
        )
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()
