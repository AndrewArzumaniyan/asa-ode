# Experiments

## Latent ODE Spiral Benchmark

Notebook:
- `experiments/neural_ode_spiral.ipynb`

Implements:
- Data generation for CW/CCW Archimedean spirals
- GRU baseline
- GRU + time-concat baseline
- Latent ODE VAE
- RMSE sweep for `nsample in {30, 50, 100}`
- Reconstruction, latent-space, interpolation, and learning-curve plots
