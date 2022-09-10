# mlp-demo

[![license][license-img]][license-url]
[![streamlit][streamlit-img]][streamlit-url]

[license-img]: https://img.shields.io/github/license/alec-hoyland/mlp-demo
[license-url]: LICENSE
[streamlit-img]: https://img.shields.io/badge/streamlit-app-blue
[streamlit-url]: https://alec-hoyland-mlp-demo-srcapp-auyr4z.streamlitapp.com/

Intuitive demonstration of a feedforward neural network for classification with a streamlit UI front-end

This repo contains the code for a 2-layer feedforward neural network
with ReLU activations and dropout
trained on the MNIST image classification task.

The MNIST dataset contains `28 x 28` grayscale images
of handwritten digits

![](https://pic3.zhimg.com/v2-60f89b96a9ab8fcacfaa735c0d3b713c_1200x500.jpg)

The model consists of a 784-dimensional input layer, followed by
two fully-connected layers of dimension 512 and 128,
followed by a 10-neuron output layer.
ReLU activation and dropout were performed between layers
and a softmax operation normalized the logits after the output layer.

A [streamlit app](https://alec-hoyland-mlp-demo-srcapp-auyr4z.streamlitapp.com/) provides an interactive demo
for testing the model on new user input.