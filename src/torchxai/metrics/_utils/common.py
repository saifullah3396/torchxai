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


def _construct_default_feature_masks(
    attributions: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    """
    Constructs feature masks from the attributions.
    Args:
        attribution (Tuple[torch.Tensor, ...]): The attributions which is a torch.Tensor or a tuple of torch.Tensors.
    Returns:
        Tuple[torch.Tensor, ...]: The feature masks corresponding to each feature in the input tensor.
    """
    last_n_features = 0
    feature_masks = []
    bsz = attributions[0].size(0)
    for attribution in attributions:
        # take each feature in attribution as a feature group and assign an increasing integer index to it
        n_features_in_attribution = attribution.view(bsz, -1).size(-1)
        feature_masks.append(
            (
                torch.arange(n_features_in_attribution, device=attribution.device)
                + last_n_features
            )
            .repeat(bsz, *([1] * len(attribution.shape[1:])))
            .view_as(attribution)
        )
        last_n_features += n_features_in_attribution

    feature_masks = tuple(feature_masks)

    _validate_feature_mask(feature_masks)
    return feature_masks


def _feature_masks_to_groups_and_counts(
    feature_masks: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function takes a tuple of feature masks and returns the grouped feature counts and the number of grouped features.

    Args:
        feature_masks (Tuple[torch.Tensor]): Tuple of feature group masks for each feature type.
        For example, for a model with 4 feature types, the input would be a tuple of 4 tensors where each tensor is of shape (batch_size, seq_length, n_features).
        And the feature type can be anything like token embeddings, position embeddings, bbox embeddings, etc.

    Returns:
        Tuple[List[torch.Tensor], List[float]]:
            The return value is a tuple of two lists. First list gives the grouped feature count which is the total
            number of elements present in each group for each feature type and each sample in the batch.
            The second list gives the total number of unique groups in each feature type and each sample.

            The output is like following:
                grouped_feature_counts = [
                    [ # feature type 1
                        [ [feature_group_1_count, feature_group_2_count, ...] # sample 1
                        [ [feature_group_1_count, feature_group_2_count, ...] # sample 2
                    ]
                ]

                n_grouped_features = [
                    [ # feature type 1
                        # N-feature-groups for sample 1
                        # N-feature-groups for sample 2
                    ]
                ]
    """

    batch_size = feature_masks[0].shape[0]
    grouped_feature_counts = []
    n_grouped_features = []
    for mask in feature_masks:
        grouped_feature_counts.append([])
        n_grouped_features.append([])
        for idx in range(batch_size):
            grouped_feature_counts[-1].append(
                torch.unique_consecutive(mask[idx, ...], return_counts=True)[1]
            )
            n_grouped_features[-1].append(len(torch.unique(mask[idx])))
        grouped_feature_counts[-1] = grouped_feature_counts[-1]
        n_grouped_features[-1] = n_grouped_features[-1]
    return grouped_feature_counts, n_grouped_features


def _generate_random_perturbation_masks(
    total_perturbations_per_feature_group: int,
    feature_masks: Tuple[torch.Tensor, ...],
    perturbation_probability: float = 0.1,
    attribution_shape: Tuple[int, ...] = None,
    device: torch.device = torch.device("cpu"),
    generator: torch.Generator = None,
) -> tuple[torch.Tensor, ...]:
    """
    Generates random perturbation masks for the input attributions. The perturbation masks are generated based on the
    feature groups present in the input attributions. The perturbation probability is the probability of perturbing a
    feature group. The perturbation masks are generated for each feature group in the input attributions and are repeated
    for each sample in the batch.

    Each group corresponds to repetitions of a single sample
    so each sample is first repeated N times and for those N repetitions we generate random perturbations based on
    the feature mask of this sample. All features belonging to the same group are perturbed together
    this means if the PAD tokens are present in the input all PAD tokens are also perturbed together so
    all of them correspond to a single feature
    Example feature mask: CLS, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, ... , 102, 102, 102, SEP, PAD, PAD, PAD # INPUT SAMPLE
    In this case the feature groups are CLS, 0, 1, 2, 3, 4, 5, ..., 102, SEP, PAD
    so by doing above we generate random perturbations for each of the feature groups as follows
    Above will generate example perturbation as:
    1. CLS, 0, PTB, 2, PTB, 4, 5, ..., 102, SEP, PTB,
    2. CLS, 0, 1, PTB, 3, 4, PTB, ..., 102, PTB, PAD,
    .
    .
    .
    10. PTB, 0, 1, PTB, 3, 4, PTB, ..., 102, SEP, PTB, where is the number of times each input sample is repeated
    """

    feature_type_shapes = tuple(x.shape[1:] for x in feature_masks)
    grouped_feature_counts, n_grouped_features = _feature_masks_to_groups_and_counts(
        feature_masks
    )
    perturbation_masks = tuple()
    for (
        n_grouped_features_per_type,
        grouped_feature_counts_per_type,
        target_expand_shape,
    ) in zip(n_grouped_features, grouped_feature_counts, feature_type_shapes):
        perturbations_per_input_type = []
        for (
            n_grouped_features_per_sample,
            grouped_feature_counts_per_sample,
        ) in zip(
            n_grouped_features_per_type,
            grouped_feature_counts_per_type,
        ):
            random_perturbation_per_sample_repetition = (
                (
                    torch.rand(
                        (
                            total_perturbations_per_feature_group,
                            n_grouped_features_per_sample,
                        ),
                        device=device,
                        generator=generator,
                    )
                    < perturbation_probability
                )
                # randomly perturb 25% of the features in the group. Each group corresponds to repetitions of a single sample
                # so each sample is first repeated N times and for those N repetitions we generate random perturbations based on
                # the feature mask of this sample. All features belonging to the same group are perturbed together
                # this means if the PAD tokens are present in the input all PAD tokens are also perturbed together so
                # all of them correspond to a single feature
                # Example feature mask: CLS, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, ... , 102, 102, 102, SEP, PAD, PAD, PAD # INPUT SAMPLE
                # In this case the feature groups are CLS, 0, 1, 2, 3, 4, 5, ..., 102, SEP, PAD
                # so by doing above we generate random perturbations for each of the feature groups as follows
                # So above will generate example perturbation as:
                # 1. CLS, 0, PTB, 2, PTB, 4, 5, ..., 102, SEP, PTB,
                # 2. CLS, 0, 1, PTB, 3, 4, PTB, ..., 102, PTB, PAD,
                # .
                # .
                # .
                # 10. PTB, 0, 1, PTB, 3, 4, PTB, ..., 102, SEP, PTB, where is the number of times each input sample is repeated
                .repeat_interleave(
                    repeats=grouped_feature_counts_per_sample, dim=1
                )  # After doing this example perturbation becomes
                # 1. CLS, 0, 0, 0, PTB, PTB, PTB, 2, PTB, 4, 5, ..., 102, 102, 102, SEP, PTB, PTB, PTB # notice this corresponds to the input sample
                # 2. CLS, 0, 0, 0, 1, 1, 1, PTB, 3, 4, PTB, ..., 102, 102, 102, PTB, PAD, PAD, PAD,
                # .
                # .
                # .
                # 10. PTB, 0, 0, 0, 1, 1, 1, PTB, 3, 4, PTB, ..., 102, 102, 102 SEP, PTB, PTB, PTB,
                .view(total_perturbations_per_feature_group, *target_expand_shape)
            )
            perturbations_per_input_type.append(
                random_perturbation_per_sample_repetition
            )
        perturbation_masks += (torch.cat(perturbations_per_input_type, dim=0),)
    return perturbation_masks


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
    return torch.cat(tuple_tensors, dim=1), tuple(x.shape for x in tuple_tensors)


def _split_tensors_to_tuple_tensors(
    tensor: Tuple[torch.Tensor], shapes, dim=1
) -> torch.Tensor:
    return tensor.split_with_sizes([x[dim] for x in shapes], dim=dim)


def _reduce_tensor_with_indices(
    source: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    source = source.flatten()
    reduced_attributions = torch.zeros_like(source)
    assert reduced_attributions.shape[0] == indices.shape[0]
    reduced_attributions.index_add_(
        0,
        indices,
        source,
    )

    # get the number of feature for each element in batch (batch is 1 anyway but this will work for multiple samples)
    n_features = (indices.max() + 1).item()

    return reduced_attributions[:n_features], n_features


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
