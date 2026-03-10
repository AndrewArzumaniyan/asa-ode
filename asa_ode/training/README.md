# Training Module

This folder contains loss computation, train/validation/test loops, metric aggregation, early stopping, and checkpoint/history persistence.

## Files

- `losses.py` — loss functions.
- `engine.py` — train/eval loops and training orchestration.

## Objective Function

For predictions $\hat{y}$, targets $y$, and observation mask $m$:

$$
\mathcal{L}_{\text{masked MSE}} = \frac{\sum (\hat{y}-y)^2 \odot m}{\max\left(\sum m, 1\right)}.
$$

Only observed entries (`mask==1`) contribute to the loss.

## `losses.py`

### `masked_mse(pred, target, mask)`

Responsibilities:

1. Compute element-wise squared error.
2. Apply observation mask.
3. Normalize by the number of observed entries.

Typical input shape: `(B, T_t_max, N)`.

Output: scalar loss tensor.

## `engine.py`

## Dataclasses

### `LoopMetrics`

Aggregated metrics for one full dataloader pass:

- `loss` — mean loss over steps.
- `step_time_sec` — average wall-clock step time.
- `peak_memory_mb` — peak accelerator memory (if backend supports it).

### `FitSummary`

Training result summary:

- `best_val_loss`
- `best_epoch`
- `checkpoint_path`

## Helper Function

### `_move_targets(batch, device)`

Moves tensors to target device and casts to `float32`:

- `target_values`
- `target_mask`

This keeps dtype/device aligned with model outputs.

## Main Loops

### `train_one_epoch(...)`

Per-batch sequence:

1. `optimizer.zero_grad(set_to_none=True)`
2. `pred = model.forward_batch(...)`
3. `loss = masked_mse(pred, target_values, target_mask)`
4. `loss.backward()`
5. Optional gradient clipping:

$$
g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|_2}\right),
$$

where $c$ is `grad_clip_norm`.

6. `optimizer.step()`
7. Update tqdm and aggregate metrics.

Returns `LoopMetrics` for the training split.

### `evaluate_model(...)`

Same data flow as training loop, but:

- runs under `@torch.no_grad()`;
- no backward pass;
- no optimizer update.

Returns `LoopMetrics` for validation/test.

### `fit_model(...)`

Top-level epoch orchestration.

Per epoch:

1. Run `train_one_epoch`.
2. Run `evaluate_model` on validation split.
3. Append epoch record to `history`.
4. If `val_loss` improves:
   - save `best_model.pt`;
   - reset patience counter.
5. If `val_loss` does not improve:
   - decrement patience;
   - trigger early stopping when `patience == 0`.

After training loop:

- save `history.json`;
- return `FitSummary`.

## Artifact Formats

### `best_model.pt`

Stored keys:

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `val_loss`

### `history.json`

List of epoch records:

- `epoch`
- `train_loss`
- `val_loss`
- `train_step_time_sec`
- `val_step_time_sec`
- `peak_memory_mb`

## Time and Memory Metrics

Runtime helpers are used:

- `reset_peak_memory(device)` before loop execution.
- `get_peak_memory_mb(device)` after loop execution.

This provides a consistent metric interface across CUDA/MPS/CPU.

## Usage Order from `scripts/train.py`

1. Build model and optimizer.
2. Run `fit_model(...)`.
3. Load best checkpoint.
4. Run `evaluate_model(...)` on test split.
5. Save final `summary.json`.
