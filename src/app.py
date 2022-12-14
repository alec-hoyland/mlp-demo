"""
Streamlit app
"""

from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

import model as nnet

st.title("Neural Network Demo")


@st.cache
def load_model() -> nnet.Net:
    """
    Loads the model from memory.

    Arguments
    ---------
    None

    Returns
    -------
    nnet.Net
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nnet.Net().to(device)
    model.load_state_dict(torch.load("src/model.pt", map_location=torch.device("cpu")))
    return model


def do_transform(x: np.ndarray) -> torch.Tensor:
    """
    Resizes input to 28x28, converts to grayscale,
    and casts as a tensor.

    Arguments
    ---------
    x : np.ndarray | torch.Tensor

    Returns
    -------
    torch.Tensor
    """

    img = Image.fromarray(x.astype("uint8"), "RGBA")
    img = ImageOps.grayscale(img)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
        ]
    )
    return transform(img)


def plot_probs(output: Iterable):
    fig = plt.figure(figsize=(3.2, 2.4))
    ax = sns.barplot(x=list(range(len(output))), y=output)
    plt.xlabel("digit")
    plt.ylabel("probability")
    plt.ylim((0, 1))
    plt.title("model output")
    return ax


def predict(input: np.ndarray, model: nnet.Net) -> Tuple[int, torch.Tensor]:
    """
    Given an image and a model, makes a prediction
    of the digit represented in the image.

    Arguments
    ---------
    input : np.ndarray
        The w x h image in grayscale

    model : nnet.Net

    Returns
    -------
    int
    """

    input = do_transform(input)

    model.eval()

    y_hat, output = model.predict(input)

    return y_hat, output

st.subheader("Model Architecture")
model_load_state = st.text("Loading PyTorch model...")
model = load_model()
# model_load_state.text("PyTorch model loaded!")
model_load_state.text(model)

st.subheader("Please draw a digit on the canvas.")
canvas_output = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=4 * 28,
    width=4 * 28,
    drawing_mode="freedraw",
    key="canvas",
    display_toolbar=True,
)

if st.button("Predict!"):
    st.subheader("Output")
    y, output = predict(canvas_output.image_data, model)
    output = output.tolist()[0]
    prob_dict = {i: np.round(output[i], decimals=2) for i in range(len(output))}
    st.pyplot(plot_probs(output).figure)
    st.write("Probabilities are:")
    st.write(prob_dict)
    st.write(f"Best-guess prediction is: {int(y)}")
