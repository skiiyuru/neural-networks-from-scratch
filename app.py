import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"# Neural Networks from Scratch"

"## 1. ReLU"


def ReLU(input):
    return np.maximum(0, input)


input = np.linspace(-10, 10, 100)

"### A single neuron"
st.markdown(
    """
Notice that:
- Bias - shifts activation point horizontally
- Weight - change slops, creates an activation or deactivation point depending on the slope
"""
)

left_col1, right_col1 = st.columns(2)

with left_col1:
    weight = st.slider("Weight1", -10.00, 10.00, 1.0, 0.1)
    bias = st.slider("Bias1", -10.00, 10.00, 0.0, 0.1)
    output1 = ReLU(input * weight + bias)

with right_col1:
    fig, ax = plt.subplots()
    ax.plot(input, output1)
    ax.set_ylim(-1, 10)
    st.pyplot(fig)

"### Two neurons"

img = Image.open("images/relu-graph2.png")
st.image(img)

st.markdown(
    """
Notice that:
- Bias of 2nd neuron - shift overall fn vertically
- Weight of 2nd neuron - if we negate, we get an area of effect (activation & deactivation points)
"""
)


for idx in range(2):
    left_col2, right_col2 = st.columns(2)

with left_col2:
    weight2 = st.slider("Weight2", -10.00, 10.00, 1.0, 0.1)
    bias2 = st.slider("Bias2", -10.00, 10.00, 0.0, 0.1)
    output2 = ReLU(input * weight2 + bias2)

    weight3 = st.slider("Weight3", -10.00, 10.00, -1.5, 0.1)
    bias3 = st.slider("Bias3", -10.00, 10.00, 4.0, 0.1)
    output3 = ReLU(output2 * weight3 + bias3)

with right_col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(input, output3)
    ax2.set_ylim(-1, 10)
    st.pyplot(fig2)

"## 2. Softmax"
"### Exponentaition: $y = e^{x}$"

exp_values = np.exp(input)
fig3, ax3 = plt.subplots()
ax3.plot(input, exp_values)
ax3.set_ylim(0, 1)
st.pyplot(fig3)

"## 3. Loss"

st.markdown(
    """
- For classification problems, we use categorical cross-entropy.
When we use one-hot encoding for our target labels, the equation
becomes:
$-log(targetClassProbability)$
- Notice that when probability is high, loss is low and vice versa.
"""
)

left_col3, right_col3 = st.columns(2)
with left_col3:
    y_pred = st.slider("Probability score", 0.1, 1.0, 0.9, 0.1)

with right_col3:
    loss = -np.log(y_pred)
    f"Loss = {loss}"
