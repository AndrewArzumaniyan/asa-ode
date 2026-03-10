# Модуль Training

Здесь реализованы loss, циклы обучения/валидации/теста, сбор метрик, ранняя остановка и сохранение checkpoint/history.

## Файлы

- `losses.py` — функции потерь.
- `engine.py` — train/eval циклы и управляющая логика обучения.

## Целевая функция

Для предсказаний $\hat{y}$, таргета $y$ и маски наблюдаемости $m$:

$$
\mathcal{L}_{\text{masked MSE}} = \frac{\sum (\hat{y}-y)^2 \odot m}{\max\left(\sum m, 1\right)}.
$$

В loss участвуют только реально наблюденные значения (`mask==1`).

## `losses.py`

### `masked_mse(pred, target, mask)`

Что делает:

1. Считает поэлементный квадрат ошибки.
2. Умножает на маску наблюдений.
3. Нормирует на число наблюденных элементов.

Типичная форма входов: `(B, T_t_max, N)`.

Выход: скалярный loss.

## `engine.py`

## Dataclass-структуры

### `LoopMetrics`

Агрегированные метрики одного прохода по dataloader:

- `loss` — средний loss по шагам.
- `step_time_sec` — среднее время одного шага.
- `peak_memory_mb` — пиковая память ускорителя (если поддерживается backend).

### `FitSummary`

Итог обучения:

- `best_val_loss`
- `best_epoch`
- `checkpoint_path`

## Вспомогательная функция

### `_move_targets(batch, device)`

Переносит на девайс и приводит к `float32`:

- `target_values`
- `target_mask`

Это гарантирует согласованность dtype с предсказаниями модели.

## Основные циклы

### `train_one_epoch(...)`

Последовательность на каждом batch:

1. `optimizer.zero_grad(set_to_none=True)`
2. `pred = model.forward_batch(...)`
3. `loss = masked_mse(pred, target_values, target_mask)`
4. `loss.backward()`
5. Опциональный gradient clipping:

$$
g \leftarrow g \cdot \min\left(1, \frac{c}{\|g\|_2}\right),
$$

где $c$ — `grad_clip_norm`.

6. `optimizer.step()`
7. Обновление tqdm и накопление статистик.

Выход: `LoopMetrics` для train-сплита.

### `evaluate_model(...)`

Повторяет data flow train-цикла, но:

- работает под `@torch.no_grad()`;
- не вызывает backward;
- не обновляет optimizer.

Выход: `LoopMetrics` для val/test.

### `fit_model(...)`

Оркестратор обучения по эпохам.

На каждой эпохе:

1. `train_one_epoch`.
2. `evaluate_model` на validation.
3. Добавление записи в `history`.
4. Если `val_loss` улучшился:
   - сохраняется `best_model.pt`;
   - patience сбрасывается.
5. Если не улучшился:
   - patience уменьшается;
   - при `patience == 0` срабатывает early stopping.

После цикла:

- сохраняется `history.json`;
- возвращается `FitSummary`.

## Формат артефактов

### `best_model.pt`

Содержит:

- `epoch`
- `model_state_dict`
- `optimizer_state_dict`
- `val_loss`

### `history.json`

Список словарей по эпохам:

- `epoch`
- `train_loss`
- `val_loss`
- `train_step_time_sec`
- `val_step_time_sec`
- `peak_memory_mb`

## Метрики времени и памяти

Используются runtime-хелперы:

- `reset_peak_memory(device)` — перед циклом.
- `get_peak_memory_mb(device)` — после цикла.

Так получается единый интерфейс отчетности для CUDA/MPS/CPU.

## Последовательность использования в `scripts/train.py`

1. Строятся модель и optimizer.
2. Запускается `fit_model(...)`.
3. Загружается лучший checkpoint.
4. Выполняется `evaluate_model(...)` на test.
5. Пишется итоговый `summary.json`.
