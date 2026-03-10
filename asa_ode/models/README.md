# Models Module

This folder contains the baseline Neural ODE implementation (without attention) for irregular time-series interpolation.

## File

- `baseline_ode.py` — all model components and the top-level `BaselineNeuralODE` class.

## Module Goal

Given patient context observations at irregular timestamps:

$$
\{(t_i, x_i, m_i)\}_{i=1}^{T_c}, \quad x_i \in \mathbb{R}^{N}, \ m_i \in \{0,1\}^{N},
$$

predict feature values at target timestamps:

$$
\{\hat{x}_j\}_{j=1}^{T_t}, \quad \hat{x}_j \in \mathbb{R}^{N}.
$$

Notation:

- $N$ — number of features.
- $M$ — latent dimension per feature.
- For a single patient, latent state shape is $(N, M)$.

## Overall Mathematical Pipeline

For one patient:

1. Context encoding (backward ODE-RNN):

$$
z_0 = \text{Encoder}(t_{1:T_c}, x_{1:T_c}, m_{1:T_c}) \in \mathbb{R}^{N \times M}.
$$

2. Latent dynamics integration at target timestamps:

$$
\frac{dz}{dt} = f_\theta(z), \quad z(t_0)=z_0,
$$

$$
z_{1:T_t} = \text{ODESolve}(f_\theta, z_0, t^{\text{target}}_{1:T_t}).
$$

3. Decoding to observation space:

$$
\hat{x}_{1:T_t} = d_\phi(z_{1:T_t}), \quad \hat{x}_{1:T_t} \in \mathbb{R}^{T_t \times N}.
$$

## Components and Class Responsibilities

### `resolve_odeint(use_adjoint: bool)`

Selects the integration backend from `torchdiffeq`:

- `odeint_adjoint` if `use_adjoint=True`.
- `odeint` if `use_adjoint=False`.

Used both in the encoder and in latent dynamics integration.

### `EncoderODEFunc`

Continuous hidden-state dynamics between two observation times inside the encoder.

Formula:

$$
\frac{dh}{dt} = f_{enc}(h),
$$

where $f_{enc}$ is an MLP:

$$
f_{enc}(h)=W_2\,\tanh(W_1 h + b_1)+b_2.
$$

Input: `(N, M)`  
Output: `(N, M)`

### `FeaturewiseODERNNEncoder`

Encodes context into `z0` by processing timestamps backward.

For reversed steps $k=1..T_c$:

1. Continuous transition over the time gap:

$$
h \leftarrow \text{ODESolve}(f_{enc}, h, [0, \Delta t_k])
$$

2. Discrete update at observation using GRU:

$$
\tilde{h} = \text{GRU}(x_k, h)
$$

3. Masked update:

$$
h \leftarrow m_k \odot \tilde{h} + (1-m_k) \odot h
$$

where $m_k \in \{0,1\}^{N \times 1}$.

Implementation detail:

- ODE-related tensors are explicitly cast to `float32` for MPS compatibility.

Inputs:

- `context_times`: `(T_c,)`
- `context_values`: `(T_c, N)`
- `context_mask`: `(T_c, N)`

Output:

- `z0`: `(N, M)`

### `LatentDynamicsFunc`

Defines baseline latent dynamics without attention.

$$
\frac{dz}{dt} = f_{dyn}(z),
$$

where $f_{dyn}$ is a shared feature-wise MLP:

$$
f_{dyn}(z)=W_2\,\tanh(W_1 z + b_1)+b_2.
$$

No explicit inter-feature attention is used in this baseline.

### `FeaturewiseDecoder`

Maps latent vectors back to scalar feature values.

$$
\hat{x} = d_\phi(z), \quad d_\phi: \mathbb{R}^{M} \to \mathbb{R}.
$$

Applied independently to each feature and each time step.

Input: `(..., N, M)`  
Output: `(..., N)`

### `BaselineNeuralODE`

Top-level model class combining encoder, latent dynamics, and decoder.

Main methods:

- `forward_single(...)` — processes one patient.
- `forward_batch(batch, device)` — processes a padded batch.

#### `forward_single` sequence

1. Cast `context_*` and `target_times` to ODE dtype (`float32`).
2. Compute `z0 = encoder(...)`.
3. Integrate latent ODE at target times.
4. Decode trajectory to predictions.

#### `forward_batch` sequence

1. Move batch tensors to selected device.
2. Iterate over patients using `context_lengths/target_lengths`.
3. Call `forward_single` per patient.
4. Write outputs into padded prediction tensor `(B, T_t_max, N)`.

## Tensor Shapes

- $B$ — batch size
- $N$ — number of features
- $M$ — latent dimension per feature
- $T_c$ — context length
- $T_t$ — target length

Padded inputs:

- `context_times`: `(B, T_c_max)`
- `context_values`: `(B, T_c_max, N)`
- `context_mask`: `(B, T_c_max, N)`
- `target_times`: `(B, T_t_max)`

Output:

- `preds`: `(B, T_t_max, N)`

## Why This Is the Baseline

This module intentionally excludes attention in $f_\theta$.  
Each feature evolves through a shared dynamics function, but without explicit pairwise attention weights.
