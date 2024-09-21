import torch
from captum.attr import visualization as viz


def visualize_image_with_explanation(
    images: torch.Tensor, explanations: torch.Tensor, explanation_method: str
) -> None:
    images = images.permute(0, 2, 3, 1) / 2 + 0.5
    images = images.cpu().detach().numpy()
    explanations = explanations.permute(0, 2, 3, 1)
    explanations = explanations.cpu().detach().numpy()

    for idx in range(len(images)):
        _ = viz.visualize_image_attr(
            None, images[idx], method="original_image", title="Original Image"
        )

        _ = viz.visualize_image_attr(
            explanations[idx],
            images[idx],
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title=f"{explanation_method}",
        )
