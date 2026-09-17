import os

import matplotlib.pyplot as plt
from PIL import Image


OUTPUT_DIR = "outputs"


def load_images():
    image_paths = {
        "Original": "original_prediction.png",
        "Conv1": "conv1_feature_maps.png",
        "Layer 2": "layer2_feature_maps.png",
        "Layer 4": "layer4_feature_maps.png",
    }

    images = {}

    for name, filename in image_paths.items():
        path = os.path.join(OUTPUT_DIR, filename)
        images[name] = Image.open(path)

    return images


def create_comparison(images):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 12)
    )

    for ax, (name, image) in zip(
        axes.flat,
        images.items()
    ):
        ax.imshow(image)
        ax.set_title(name, fontsize=16)
        ax.axis("off")

    fig.suptitle(
        "Watching a Neural Network Process an Image",
        fontsize=20
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "layer_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


def main():
    images = load_images()
    create_comparison(images)


if __name__ == "__main__":
    main()