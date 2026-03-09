from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


EXCLUDED_PARAMETERS = {"RecordID"}


@dataclass
class DataBundle:
    """Stores train/val/test dataloaders and dataset metadata."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    feature_names: list[str]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor


def parse_time_to_hours(time_str: str) -> float:
    """Parses PhysioNet time string into floating point hours."""
    hh, mm = time_str.strip().split(":")
    return float(int(hh) + int(mm) / 60.0)


def collect_patient_files(data_root: str | Path) -> list[Path]:
    """Collects all patient files recursively from data root."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")
    files = sorted([p for p in root.rglob("*.txt") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .txt patient files found under: {root}")
    return files


def infer_feature_names(files: list[Path], explicit_feature_names: list[str] | None) -> list[str]:
    """Infers feature names from data files or returns user-provided list."""
    if explicit_feature_names is not None:
        return list(explicit_feature_names)

    features: set[str] = set()
    for file_path in tqdm(files, desc="Infer features", leave=False):
        with file_path.open("r", encoding="utf-8") as fp:
            lines = fp.readlines()
        for raw in lines[1:]:
            parts = [item.strip() for item in raw.split(",")]
            if len(parts) < 3:
                continue
            param = parts[1]
            if param in EXCLUDED_PARAMETERS:
                continue
            try:
                float(parts[2])
            except ValueError:
                continue
            features.add(param)

    if not features:
        raise ValueError("Could not infer any numeric feature from dataset.")
    return sorted(features)


def parse_patient_file(file_path: Path, feature_to_idx: dict[str, int]) -> dict[str, torch.Tensor | str] | None:
    """Parses one patient file into tensors of times, values and masks."""
    with file_path.open("r", encoding="utf-8") as fp:
        lines = fp.readlines()
    if len(lines) <= 1:
        return None

    events: list[tuple[float, str, float]] = []
    for raw in lines[1:]:
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) < 3:
            continue
        time_str, param, value_str = parts[0], parts[1], parts[2]
        if param in EXCLUDED_PARAMETERS:
            continue
        if param not in feature_to_idx:
            continue
        try:
            t = parse_time_to_hours(time_str)
            v = float(value_str)
        except ValueError:
            continue
        events.append((t, param, v))

    if not events:
        return None

    times_sorted = sorted(set(event[0] for event in events))
    time_to_index = {t: i for i, t in enumerate(times_sorted)}

    t_len = len(times_sorted)
    n_feat = len(feature_to_idx)
    values = torch.zeros((t_len, n_feat), dtype=torch.float32)
    mask = torch.zeros((t_len, n_feat), dtype=torch.float32)

    for t, param, value in events:
        ti = time_to_index[t]
        fi = feature_to_idx[param]
        values[ti, fi] = float(value)
        mask[ti, fi] = 1.0

    times = torch.tensor(times_sorted, dtype=torch.float32)
    patient_id = file_path.stem
    return {
        "patient_id": patient_id,
        "times": times,
        "values": values,
        "mask": mask,
    }


def build_cache(
    data_root: str | Path,
    cache_path: str | Path,
    feature_names: list[str] | None,
    min_timepoints: int,
) -> dict[str, object]:
    """Builds cached dataset tensors from raw PhysioNet files."""
    files = collect_patient_files(data_root)
    inferred_features = infer_feature_names(files, feature_names)
    feature_to_idx = {name: idx for idx, name in enumerate(inferred_features)}

    samples: list[dict[str, torch.Tensor | str]] = []
    for file_path in tqdm(files, desc="Parse patients"):
        sample = parse_patient_file(file_path, feature_to_idx)
        if sample is None:
            continue
        times = sample["times"]
        if isinstance(times, torch.Tensor) and times.numel() >= min_timepoints:
            samples.append(sample)

    if not samples:
        raise ValueError("No valid patients after parsing and filtering.")

    cache = {
        "samples": samples,
        "feature_names": inferred_features,
    }

    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_file)
    return cache


def load_or_build_cache(
    data_root: str | Path,
    cache_path: str | Path,
    feature_names: list[str] | None,
    min_timepoints: int,
    rebuild_cache: bool,
) -> dict[str, object]:
    """Loads existing cache or rebuilds it from raw files."""
    cache_file = Path(cache_path)
    if cache_file.exists() and not rebuild_cache:
        return torch.load(cache_file)
    return build_cache(
        data_root=data_root,
        cache_path=cache_path,
        feature_names=feature_names,
        min_timepoints=min_timepoints,
    )


def split_indices(total_size: int, train_ratio: float, val_ratio: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Splits sample indices into train, validation and test subsets."""
    if total_size < 3:
        raise ValueError("Dataset needs at least 3 samples for train/val/test split.")

    idx = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(idx)

    train_size = max(1, int(round(total_size * train_ratio)))
    val_size = max(1, int(round(total_size * val_ratio)))
    test_size = total_size - train_size - val_size

    while test_size < 1:
        if train_size >= val_size and train_size > 1:
            train_size -= 1
        elif val_size > 1:
            val_size -= 1
        else:
            break
        test_size = total_size - train_size - val_size

    if test_size < 1:
        raise ValueError("Split produced an empty subset. Adjust split ratios or dataset size.")

    train_end = train_size
    val_end = train_size + val_size

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:val_end + test_size]

    if not train_idx or not val_idx or not test_idx:
        raise ValueError("Split produced an empty subset. Adjust split ratios.")

    return train_idx, val_idx, test_idx


def compute_feature_stats(samples: list[dict[str, torch.Tensor | str]], indices: list[int], num_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes per-feature mean and std from observed train values only."""
    sum_v = torch.zeros(num_features, dtype=torch.float64)
    sq_sum_v = torch.zeros(num_features, dtype=torch.float64)
    count_v = torch.zeros(num_features, dtype=torch.float64)

    for i in tqdm(indices, desc="Feature stats", leave=False):
        sample = samples[i]
        values = sample["values"]
        mask = sample["mask"]
        if not isinstance(values, torch.Tensor) or not isinstance(mask, torch.Tensor):
            continue
        sum_v += (values.double() * mask.double()).sum(dim=0)
        sq_sum_v += ((values.double() ** 2) * mask.double()).sum(dim=0)
        count_v += mask.double().sum(dim=0)

    safe_count = torch.clamp(count_v, min=1.0)
    mean = sum_v / safe_count
    var = sq_sum_v / safe_count - mean ** 2
    var = torch.clamp(var, min=1e-6)
    std = torch.sqrt(var)

    missing = count_v == 0
    mean[missing] = 0.0
    std[missing] = 1.0

    return mean.float(), std.float()


def normalize_values(values: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Applies feature-wise normalization only on observed entries."""
    return ((values - mean.unsqueeze(0)) / std.unsqueeze(0)) * mask


class PhysioNetDataset(Dataset):
    """Creates context-target samples for irregular time series interpolation."""

    def __init__(
        self,
        samples: list[dict[str, torch.Tensor | str]],
        indices: list[int],
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
    ) -> None:
        self.samples = samples
        self.indices = indices
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self) -> int:
        """Returns number of patient samples in this split."""
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        """Returns one patient split into context and target segments."""
        sample = self.samples[self.indices[item]]

        times = sample["times"]
        values = sample["values"]
        mask = sample["mask"]
        patient_id = sample["patient_id"]

        if not isinstance(times, torch.Tensor) or not isinstance(values, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("Sample tensors are corrupted.")

        t_len = times.numel()
        split_idx = max(1, t_len // 2)
        if split_idx >= t_len:
            split_idx = t_len - 1

        context_times = times[:split_idx]
        target_times = times[split_idx:]

        origin = context_times[0]
        context_times = context_times - origin
        target_times = target_times - origin

        context_values = normalize_values(values[:split_idx], mask[:split_idx], self.feature_mean, self.feature_std)
        target_values = normalize_values(values[split_idx:], mask[split_idx:], self.feature_mean, self.feature_std)

        context_mask = mask[:split_idx]
        target_mask = mask[split_idx:]

        return {
            "patient_id": patient_id,
            "context_times": context_times,
            "context_values": context_values,
            "context_mask": context_mask,
            "target_times": target_times,
            "target_values": target_values,
            "target_mask": target_mask,
        }


def collate_physionet_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    """Pads variable-length patient samples into a single batch dictionary."""
    bsz = len(batch)
    patient_ids = [str(item["patient_id"]) for item in batch]

    num_features = batch[0]["context_values"].shape[-1]
    max_context_len = max(item["context_times"].numel() for item in batch)
    max_target_len = max(item["target_times"].numel() for item in batch)

    context_times = torch.zeros((bsz, max_context_len), dtype=torch.float32)
    context_values = torch.zeros((bsz, max_context_len, num_features), dtype=torch.float32)
    context_mask = torch.zeros((bsz, max_context_len, num_features), dtype=torch.float32)

    target_times = torch.zeros((bsz, max_target_len), dtype=torch.float32)
    target_values = torch.zeros((bsz, max_target_len, num_features), dtype=torch.float32)
    target_mask = torch.zeros((bsz, max_target_len, num_features), dtype=torch.float32)

    context_lengths = torch.zeros((bsz,), dtype=torch.long)
    target_lengths = torch.zeros((bsz,), dtype=torch.long)

    for i, item in enumerate(batch):
        c_len = item["context_times"].numel()
        t_len = item["target_times"].numel()

        context_lengths[i] = c_len
        target_lengths[i] = t_len

        context_times[i, :c_len] = item["context_times"]
        context_values[i, :c_len] = item["context_values"]
        context_mask[i, :c_len] = item["context_mask"]

        target_times[i, :t_len] = item["target_times"]
        target_values[i, :t_len] = item["target_values"]
        target_mask[i, :t_len] = item["target_mask"]

    return {
        "patient_ids": patient_ids,
        "context_times": context_times,
        "context_values": context_values,
        "context_mask": context_mask,
        "context_lengths": context_lengths,
        "target_times": target_times,
        "target_values": target_values,
        "target_mask": target_mask,
        "target_lengths": target_lengths,
    }


def build_dataloaders(
    data_root: str | Path,
    cache_path: str | Path,
    batch_size_train: int,
    batch_size_eval: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    feature_names: list[str] | None,
    min_timepoints: int,
    rebuild_cache: bool = False,
) -> DataBundle:
    """Builds train/val/test dataloaders with cached preprocessing."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    cache = load_or_build_cache(
        data_root=data_root,
        cache_path=cache_path,
        feature_names=feature_names,
        min_timepoints=min_timepoints,
        rebuild_cache=rebuild_cache,
    )

    samples = cache["samples"]
    feature_list = cache["feature_names"]
    if not isinstance(samples, list) or not isinstance(feature_list, list):
        raise TypeError("Cache format is invalid.")

    train_idx, val_idx, test_idx = split_indices(
        total_size=len(samples),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    feature_mean, feature_std = compute_feature_stats(samples, train_idx, len(feature_list))

    train_dataset = PhysioNetDataset(samples, train_idx, feature_mean, feature_std)
    val_dataset = PhysioNetDataset(samples, val_idx, feature_mean, feature_std)
    test_dataset = PhysioNetDataset(samples, test_idx, feature_mean, feature_std)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_physionet_batch,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_physionet_batch,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_physionet_batch,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        feature_names=feature_list,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
