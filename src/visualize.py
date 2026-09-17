import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from model import load_model
from hooks import FeatureExtractor


IMAGE_PATH = "sample.avif"
OUTPUT_DIR = "outputs"


def prepare_image(image_path, weights):
    image = Image.open(image_path).convert("RGB")

    transform = weights.transforms()
    tensor = transform(image).unsqueeze(0)

    return image, tensor


def visualize_feature_maps(features, layer_name, num_maps=16):
    feature_maps = features[layer_name][0]

    num_maps = min(num_maps, feature_maps.shape[0])

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for index, ax in enumerate(axes.flat):
        ax.imshow(feature_maps[index].cpu(), cmap="viridis")
        ax.set_title(f"Map {index + 1}")
        ax.axis("off")

    fig.suptitle(
        f"{layer_name} — First {num_maps} Feature Maps",
        fontsize=16
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{layer_name}_feature_maps.png"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, weights = load_model()

    image, tensor = prepare_image(IMAGE_PATH, weights)

    layers = [
        "conv1",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ]

    extractor = FeatureExtractor(model, layers)

    with torch.no_grad():
        output = model(tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_probability, top_class = probabilities.max(dim=0)

    categories = weights.meta["categories"]
    prediction = categories[top_class.item()]
    confidence = top_probability.item() * 100

    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(
        f"{prediction}\n"
        f"Confidence: {confidence:.2f}%"
    )
    plt.axis("off")

    original_path = os.path.join(
        OUTPUT_DIR,
        "original_prediction.png"
    )

    plt.savefig(original_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {original_path}")

    visualize_feature_maps(
        extractor.features,
        "conv1"
    )

    visualize_feature_maps(
        extractor.features,
        "layer2"
    )

    visualize_feature_maps(
        extractor.features,
        "layer4"
    )

    extractor.remove_hooks()


if __name__ == "__main__":
    main()