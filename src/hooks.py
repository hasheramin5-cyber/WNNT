import torch


class FeatureExtractor:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.features = {}
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        for name, layer in self.model.named_modules():
            if name in self.layers:
                hook = layer.register_forward_hook(
                    self._create_hook(name)
                )
                self.hooks.append(hook)

    def _create_hook(self, name):
        def hook(module, inputs, output):
            self.features[name] = output.detach()

        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()


if __name__ == "__main__":
    from model import load_model

    model, _ = load_model()

    layers = [
        "conv1",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ]

    extractor = FeatureExtractor(model, layers)

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        model(dummy_input)

    print("Captured feature maps:")

    for name, feature in extractor.features.items():
        print(f"{name}: {tuple(feature.shape)}")

    extractor.remove_hooks()