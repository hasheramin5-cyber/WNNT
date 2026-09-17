# Can We Watch a Neural Network Think?

A visual experiment to explore what happens inside a neural network as an image passes through its layers.

The project extracts intermediate feature maps from a convolutional neural network and visualizes how the network transforms an image before producing a prediction.

## Goal

Instead of treating a neural network as a black box:

```text
Image → Model → Prediction
```

we visualize what happens inside:

```text
Image
  ↓
Early Features
  ↓
Edges and Textures
  ↓
Patterns
  ↓
High-Level Features
  ↓
Prediction
```

## Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

## Experiment

The experiment uses a pretrained ResNet18 model to visualize intermediate activations from different layers.

The test image was classified as:

**Toucan → 99.92% confidence**

Generated visualizations are available in the [`outputs/`](outputs/) directory.

## Article

Read the full experiment and visualization on Medium:

[Can We Watch a Neural Network Think?](https://medium.com/@hasheramin/can-we-watch-a-neural-network-think-79d0d04969bb)