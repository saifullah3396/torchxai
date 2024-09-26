from captum.attr import visualization as viz


def visualize_attribution(image, attribution, title="Attributions"):
    image = image.view(1, 28, 28).permute(1, 2, 0)
    attribution = attribution.view(1, 28, 28).permute(1, 2, 0)
    image = image.detach().cpu().numpy()
    attribution = attribution.detach().cpu().numpy()
    _ = viz.visualize_image_attr(
        attribution,
        image,
        method="blended_heat_map",
        sign="all",
        show_colorbar=True,
        title=title,
    )
