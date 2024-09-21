from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers.tokenization_utils import PreTrainedTokenizer


def summarize_attribution(attributions: torch.Tensor) -> torch.Tensor:
    attributions = attributions
    attributions = attributions / torch.norm(attributions)
    return attributions


def summarize_attributions(attributions: torch.Tensor) -> torch.Tensor:
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def visualize_token_explanations(
    explanation: torch.Tensor, token_strings: List[str], explanation_key: str
) -> None:
    assert explanation.shape[0] == len(token_strings)
    explanation = summarize_attribution(explanation)

    fig, ax = plt.subplots(figsize=(25, 5))
    pad_token_start_idx = token_strings.index("<pad>")
    ax.bar(token_strings[:pad_token_start_idx], explanation[:pad_token_start_idx])

    plt.xlabel("Tokens")
    plt.ylabel("Explanation")
    plt.title(explanation_key)
    plt.xticks(rotation=90)
    plt.show()


def visualize_image_explanations(
    explanation: torch.Tensor, explanation_key: str
) -> None:
    explanation = summarize_attribution(explanation)

    dim = int(np.sqrt(explanation.shape[0]))
    explanation = explanation.view(dim, dim)

    fig, ax = plt.subplots(figsize=(25, 5))
    ax.imshow(explanation)

    plt.xlabel("Tokens")
    plt.ylabel("Explanation")
    plt.title(explanation_key)
    plt.xticks(rotation=90)
    plt.show()


def convert_token_ids_to_strings(
    tokenizer: PreTrainedTokenizer, input_ids: torch.Tensor
) -> List[str]:
    token_strings = []
    for input_id in input_ids:
        token_strings.append(tokenizer.convert_ids_to_tokens(input_id))
    return token_strings
