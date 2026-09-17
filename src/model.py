import torch
from torchvision.models import resnet18, ResNet18_Weights


def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    model.eval()

    return model, weights


if __name__ == "__main__":
    model, weights = load_model()

    print("Model loaded successfully.")
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of layers: {len(list(model.children()))}")