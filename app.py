import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

"# Neural Networks from Scratch"

"## 1. ReLU"


def ReLU(input):
    return np.maximum(0, input)


left_col, right_col = st.columns(2)

with left_col:
    weight = st.slider("Weight", -10, 10, 1)
    bias = st.slider("Bias", -10, 10, 1)
    input = np.linspace(-10, 10, 100)
    output = ReLU(input * weight + bias)

with right_col:
    fig, ax = plt.subplots()
    ax.plot(input, output)
    ax.set_ylim(-1, 10)
    st.pyplot(fig)
