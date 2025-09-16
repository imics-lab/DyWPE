## Dynamic Wavelet Positional Encoding (DyWPE): A Signal-Aware Framework


###  Mathematical Formulation

Let the input be a batch of multivariate time series $X \in \mathbb{R}^{B \times L \times d_x}$, where $B$ is the batch size, $L$ is the sequence length, and $d_x$ is the number of input channels. The objective is to produce a positional encoding $P_{DyWPE} \in \mathbb{R}^{B \times L \times d_{model}}$, where $d_{model}$ is the transformer's hidden dimension.

The process consists of five key steps for each instance $x \in \mathbb{R}^{L \times d_x}$ in the batch.

### Step 1: Input Projection

First, the input time series features are projected into the model's hidden dimension, $d_{model}$.
$$
X_{proj} = X \cdot W_{proj} + b_{proj}
$$
where $W_{proj} \in \mathbb{R}^{d_x \times d_{model}}$ and $b_{proj} \in \mathbb{R}^{d_{model}}$ are learnable parameters.

### Step 2: Multi-Level Wavelet Decomposition (DWT)

The core of DyWPE is the analysis of the **original input signal** $x$. To handle multivariate signals, we first project the $d_x$ channels to a single representative channel for wavelet analysis.
$$
x_{mono} = x \cdot w_{channel}
$$
where $w_{channel} \in \mathbb{R}^{d_x}$ is a learnable projection vector.

Next, we apply a $J$-level 1D Discrete Wavelet Transform (DWT) to this single-channel signal $x_{mono} \in \mathbb{R}^{L}$.
$$
(cA_J, [cD_J, cD_{J-1}, ..., cD_1]) = \text{DWT}(x_{mono})
$$
This decomposition yields:
-   **$cA_J$**: Approximation coefficients at the coarsest level $J$, capturing the low-frequency, large-scale trends of the signal.
-   **$cD_j$**: A set of detail coefficients for each level $j \in [1, J]$, capturing the high-frequency, fine-scale details.

### Step 3: Learnable Scale Embeddings and Dynamic Modulation

We introduce a set of learnable embedding vectors, which act as "prototypes" for each temporal scale captured by the DWT.
$$
E_{scales} = \{e_{A_J}, e_{D_J}, e_{D_{J-1}}, ..., e_{D_1}\}
$$
where each embedding vector $e \in \mathbb{R}^{d_{model}}$. There is one embedding for the approximation level and one for each of the $J$ detail levels.

The key innovation of DyWPE is **dynamic modulation**. We use the signal's actual wavelet coefficients to modulate these learnable scale embeddings via a learnable gating mechanism. For each scale's coefficient tensor $c$ and its corresponding scale embedding $e$:
$$
\text{gate}(e, c) = \left( \sigma(W_g e) \odot \tanh(W_v e) \right) \otimes c'
$$
where:
-   $W_g, W_v \in \mathbb{R}^{d_{model} \times d_{model}}$ are learnable weight matrices.
-   $\sigma$ is the sigmoid function, acting as a soft gate.
-   $\odot$ denotes element-wise multiplication.
-   $c'$ is the coefficient tensor broadcasted to shape `(B, L_coeffs, d_model)`.
-   $\otimes$ denotes element-wise multiplication.

This process is applied to all approximation and detail coefficients, resulting in modulated coefficient tensors:
$$
Yl_{mod} = \text{gate}(e_{A_J}, cA_J)
$$
$$
Yh_{mod} = [\text{gate}(e_{D_J}, cD_J), ..., \text{gate}(e_{D_1}, cD_1)]
$$

### Step 4: Reconstruction via Inverse DWT (IDWT)

We reconstruct the final positional encoding by applying the Inverse DWT (IDWT) to the modulated coefficient tensors.
$$
P_{DyWPE} = \text{IDWT}(Yl_{mod}, Yh_{mod})
$$
The IDWT synthesizes the modulated, multi-scale information back into a single sequence of length $L$. The resulting $P_{DyWPE} \in \mathbb{R}^{B \times L \times d_{model}}$ is a rich, signal-aware positional encoding.

### Step 5: Final Combination

Finally, the generated positional encoding is added to the projected input embeddings.
$$
X_{final} = X_{proj} + P_{DyWPE}
$$
This final tensor $X_{final}$ is then fed into the subsequent layers of the transformer.



