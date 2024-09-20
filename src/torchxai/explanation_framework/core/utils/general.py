from __future__ import annotations

import hashlib
from typing import Any, List, Tuple

import pandas as pd
import torch
from attr import dataclass
from sklearn.metrics import classification_report
from torchfusion.core.utilities.logging import get_logger

from torchxai import *  # noqa

logger = get_logger(__name__)


@dataclass
class ExplanationParameters:
    model_inputs: Tuple[torch.Tensor, ...]
    baselines: Tuple[torch.Tensor, ...]
    feature_mask: Tuple[torch.Tensor, ...]
    additional_forward_args: Tuple[Any]


def pretty_classification_report(y_true: List[int], y_pred: List[int]) -> pd.DataFrame:
    """
    Generate a pretty classification report.
    Parameters:
    - y_true (array-like): The true labels.
    - y_pred (array-like): The predicted labels.
    Returns:
    - report_df (pandas DataFrame): A DataFrame containing the classification report with rounded values.
    Example:
    ```
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred = [0, 1, 1, 1]
    >>> report_df = pretty_classification_report(y_true, y_pred)
    >>> print(report_df)
                    precision  recall  f1-score  support
    0                1.00    0.50      0.67     2.00
    1                0.67    1.00      0.80     2.00
    accuracy         0.75    0.75      0.75     4.00
    macro avg        0.83    0.75      0.73     4.00
    weighted avg     0.83    0.75      0.73     4.00
    ```
    """

    # Generate the classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    # Convert the dictionary report to a pandas DataFrame for pretty printing
    report_df = pd.DataFrame(report).transpose()

    # Display the DataFrame with rounded values for easier reading
    report_df = report_df.round(2)

    logger.info(f"Classification Report:\n{report_df}")
    return report_df


def generate_unique_sample_key(
    sample_index: int,
    image_file_path: str,
):
    # generate unique identifier for this sample using index and image file path
    file_hash = hashlib.md5(image_file_path.encode()).hexdigest()
    sample_key = f"{sample_index}_{file_hash}"
    return sample_key


def convert_numeric_ner_labels_to_ner_tags(
    class_labels: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor
) -> Tuple[List[List[str]], List[List[str]]]:
    # convert tensor to list
    preds = preds.detach().cpu().tolist()
    true_preds = [
        [class_labels[p] for (p, l) in zip(prediction, target) if l != -100]
        for prediction, target in zip(preds, targets)
    ]

    # convert tensor to list
    targets = targets.detach().cpu().tolist()
    true_targets = [
        [class_labels[l] for (p, l) in zip(prediction, target) if l != -100]
        for prediction, target in zip(preds, targets)
    ]

    return true_preds, true_targets


def expand_feature_mask_to_inputs(
    feature_mask: Tuple[torch.Tensor], inputs: Tuple[torch.Tensor]
) -> Tuple[torch.Tensor]:
    return tuple(
        (
            mask.unsqueeze(-1).expand_as(input)
            if len(mask.shape) < len(input.shape)
            else mask.expand_as(input)
        )
        for input, mask in zip(inputs, feature_mask)
    )


def perturb_fn_drop_batched_single_output(
    grouped_feature_counts,
    n_grouped_features,
    drop_probability=0.25,
    drop_feature_type=None,
    debugging=False,
):
    def wrapped(inputs, baselines):
        # to compute infidelity we take randomly set half the features to baseline
        # here we generate random indices which will be set to baseline
        # input shape should be (batch_size, seq_length, feature_dim)
        # first we generate rand boolean vectors of size (batch_size, seq_length)
        # then we repeat each bool value n times where n is the number of features in the group given by "repeats"
        # then the input expanded to feature_dim
        # Note: This happens even in cases where features are not grouped together because we want the token
        # removal frequency to be the same among all attribution methods whether it uses grouped features or not
        # for example for deep_lift method the feature groups=n_tokens + 1 (CLS) + 1 (SEP) + n_pad_tokens (PAD)
        # but for lime method the feature groups=n_words (each word consists of m tokens) + 1 (CLS) + 1 (SEP) + n_pad_tokens (PAD)

        total_samples = len(n_grouped_features[0])
        drops = tuple()
        for input, n_grouped_features_per_type, grouped_feature_counts_per_type in zip(
            inputs, n_grouped_features, grouped_feature_counts
        ):

            drops_per_input_type = []
            for (
                repeated_input,
                n_grouped_features_per_sample,
                grouped_feature_counts_per_sample,
            ) in zip(
                input.chunk(total_samples, dim=0),
                n_grouped_features_per_type,
                grouped_feature_counts_per_type,
            ):
                random_dropout_per_sample_repetition = (
                    (
                        torch.rand(
                            (repeated_input.shape[0], n_grouped_features_per_sample),
                            device=input.device,
                        )
                        < drop_probability
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
                    .unsqueeze(
                        -1
                    ).expand(  # add a new dimension at the end so that now each sequence element is removed as a whole in the feature dimension which is usually 768 for transformers
                        repeated_input.shape
                    )  # expand the dimension to the original input shape so from (batch_size, seq_length) to (batch_size, seq_length, n_features)
                )
                drops_per_input_type.append(random_dropout_per_sample_repetition)
            drops += (torch.cat(drops_per_input_type, dim=0),)

        drops, inputs_perturbed = tuple(x.float() for x in drops), tuple(
            input * ~drop_indices
            + baseline
            * drop_indices  # this is the actual perturbation where we set the features to baseline if they are dropped
            for input, baseline, drop_indices in zip(inputs, baselines, drops)
        )

        if debugging:
            import matplotlib.pyplot as plt

            for drop, ptb in zip(drops, inputs_perturbed):
                fig, axes = plt.subplots(figsize=(50, 10), nrows=2)
                axes[0].matshow(drop[:, :, 0][:, :50].cpu().numpy())
                axes[1].matshow(ptb[:, :, 0][:, :50].cpu().numpy())
                plt.show()

        return drops, inputs_perturbed

    return wrapped


def unpack_explanation_parameters(
    explanation_parameters: ExplanationParameters,
) -> Tuple[torch.Tensor, ...]:
    if isinstance(explanation_parameters.model_inputs, dict):
        assert isinstance(explanation_parameters.baselines, dict)
        assert (
            explanation_parameters.model_inputs.keys()
            == explanation_parameters.baselines.keys()
        )
        assert (
            explanation_parameters.model_inputs.keys()
            == explanation_parameters.feature_mask.keys()
        )

        inputs = tuple(explanation_parameters.model_inputs.values())
        baselines = tuple(explanation_parameters.baselines.values())
        feature_mask = tuple(explanation_parameters.feature_mask.values())
        additional_forward_args = explanation_parameters.additional_forward_args
    else:
        inputs = explanation_parameters.model_inputs
        baselines = explanation_parameters.baselines
        feature_mask = explanation_parameters.feature_mask
        additional_forward_args = explanation_parameters.additional_forward_args

    return inputs, baselines, feature_mask, additional_forward_args


def grid_segmenter(images: torch.Tensor, cell_size: int = 16) -> torch.Tensor:
    feature_mask = []
    for image in images:
        # image dimensions are C x H x H
        dim_x, dim_y = image.shape[1] // cell_size, image.shape[2] // cell_size
        mask = (
            torch.arange(dim_x * dim_y)
            .view((dim_x, dim_y))
            .repeat_interleave(cell_size, dim=0)
            .repeat_interleave(cell_size, dim=1)
            .long()
            .unsqueeze(0)
        )
        feature_mask.append(mask)
    return torch.stack(feature_mask)
