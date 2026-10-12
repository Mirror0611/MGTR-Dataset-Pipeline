from __future__ import annotations

import csv
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn import metrics as sk_metrics
from torch.utils.data import Dataset
from tqdm import tqdm


BIN_EDGES = [24, 48, 72, 96, 120, 144, 168, 192, 336]
N_BINS = len(BIN_EDGES) + 1

FEATURE_CACHE_VERSION = 1
VALUE_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class LOSSample:
    stay: str
    period_length: float
    y_true: float


def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_listfile(path: str) -> List[LOSSample]:
    samples: List[LOSSample] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                LOSSample(
                    stay=row["stay"],
                    period_length=float(row["period_length"]),
                    y_true=float(row["y_true"]),
                )
            )
    return samples


def extract_patient_id(stay: str) -> str:
    pos = stay.find("_")
    return stay if pos < 0 else stay[:pos]


def hours_to_bin(y_hours: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.searchsorted(BIN_EDGES, np.asarray(y_hours, dtype=np.float32))


def build_class_weights(
    y_bins: np.ndarray,
    n_classes: int = N_BINS,
    power: float = 0.5,
    max_weight: float = 5.0,
) -> torch.Tensor:
    counts = np.bincount(y_bins, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / np.power(counts, power)
    weights = weights / weights.sum() * n_classes
    weights = np.clip(weights, 1.0 / max_weight, max_weight)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_stay_inverse_sample_weights(
    samples: Sequence[LOSSample],
    indices: Sequence[int],
    power: float = 1.0,
) -> torch.DoubleTensor:
    stay_counts: Dict[str, int] = {}
    for i in indices:
        stay_counts[samples[i].stay] = stay_counts.get(samples[i].stay, 0) + 1
    weights = [1.0 / (stay_counts[samples[i].stay] ** power) for i in indices]
    return torch.as_tensor(weights, dtype=torch.double)


def patient_split_indices(
    samples: Sequence[LOSSample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    pids = [extract_patient_id(s.stay) for s in samples]
    unique_pids = sorted(set(pids))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_pids)
    n_val = int(len(unique_pids) * val_ratio)
    val_pids = set(unique_pids[:n_val])

    train_idx: List[int] = []
    val_idx: List[int] = []
    for i, pid in enumerate(pids):
        if pid in val_pids:
            val_idx.append(i)
        else:
            train_idx.append(i)
    print(
        f"Patient split: patients={len(unique_pids)} "
        f"train={len(train_idx)} val={len(val_idx)}"
    )
    return train_idx, val_idx


def maybe_subsample_indices(
    indices: Sequence[int],
    ratio: float,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> List[int]:
    out = list(indices)
    rng = np.random.RandomState(seed)
    if ratio < 1.0:
        n = max(1, int(len(out) * ratio))
        out = rng.choice(out, size=n, replace=False).tolist()
    if max_samples is not None and max_samples > 0 and len(out) > max_samples:
        out = rng.choice(out, size=max_samples, replace=False).tolist()
    return out


def parse_measurement_value(value: str) -> float:
    value = (value or "").strip()
    if not value or value.lower() == "nan":
        return math.nan
    try:
        return float(value)
    except ValueError:
        match = VALUE_RE.search(value)
        return float(match.group(0)) if match else math.nan


def read_header_from_split(data_dir: str, split_name: str) -> List[str]:
    split_dir = os.path.join(data_dir, split_name)
    for name in os.listdir(split_dir):
        if name.endswith("_timeseries.csv"):
            with open(os.path.join(split_dir, name), "r", encoding="utf-8") as f:
                return next(csv.reader(f))[1:]
    raise FileNotFoundError(f"No timeseries CSV found under {split_dir}")


def read_timeseries_csv(path: str, expected_features: Optional[Sequence[str]] = None):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        features = header[1:]
        if expected_features is not None and list(features) != list(expected_features):
            raise ValueError(f"Unexpected header in {path}")

        times: List[float] = []
        values: List[List[float]] = []
        masks: List[List[bool]] = []
        for row in reader:
            if not row:
                continue
            try:
                hour = float(row[0])
            except ValueError:
                continue

            vals: List[float] = []
            obs: List[bool] = []
            for v in row[1 : 1 + len(features)]:
                parsed = parse_measurement_value(v)
                if math.isnan(parsed):
                    vals.append(0.0)
                    obs.append(False)
                else:
                    vals.append(parsed)
                    obs.append(True)
            times.append(hour)
            values.append(vals)
            masks.append(obs)

    if not times:
        return (
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, len(expected_features or []), dtype=torch.float32),
            torch.empty(0, len(expected_features or []), dtype=torch.bool),
        )

    order = np.argsort(np.asarray(times, dtype=np.float32))
    return (
        torch.tensor(np.asarray(times, dtype=np.float32)[order], dtype=torch.float32),
        torch.tensor(np.asarray(values, dtype=np.float32)[order], dtype=torch.float32),
        torch.tensor(np.asarray(masks, dtype=np.bool_)[order], dtype=torch.bool),
    )


class StayTimeSeriesStore:
    """Stay-level cache for raw MIMIC benchmark time-series files."""

    def __init__(
        self,
        data_dir: str,
        split_name: str,
        samples: Sequence[LOSSample],
        cache_dir: str = "./processed_los_structured",
        feature_names: Optional[Sequence[str]] = None,
        rebuild_cache: bool = False,
        max_delta_hours: float = 48.0,
        max_age_days: float = 30.0,
    ):
        self.data_dir = data_dir
        self.split_name = split_name
        self.samples = samples
        self.cache_dir = cache_dir
        self.max_delta_hours = float(max_delta_hours)
        self.max_age_days = float(max_age_days)
        os.makedirs(cache_dir, exist_ok=True)

        self.feature_names = list(feature_names) if feature_names else read_header_from_split(data_dir, split_name)
        self.raw_cache_path = os.path.join(cache_dir, f"{split_name}_stay_raw.pt")
        self.raw = self._load_or_build_raw(rebuild_cache)
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.prepared: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def input_dim(self) -> int:
        # normalized value, observation mask, time-since-last-observed, age-to-current-period
        return self.n_features * 3 + 1

    def _load_or_build_raw(self, rebuild_cache: bool):
        if os.path.exists(self.raw_cache_path) and not rebuild_cache:
            obj = safe_torch_load(self.raw_cache_path)
            if (
                obj.get("version") == FEATURE_CACHE_VERSION
                and obj.get("feature_names") == self.feature_names
            ):
                print(f"Loaded structured raw cache: {self.raw_cache_path}")
                return obj["stays"]
            print(f"Ignoring stale structured cache: {self.raw_cache_path}")

        split_dir = os.path.join(self.data_dir, self.split_name)
        unique_stays = sorted(set(s.stay for s in self.samples))
        raw = {}
        for stay in tqdm(unique_stays, desc=f"Reading {self.split_name} stays"):
            path = os.path.join(split_dir, stay)
            times, values, masks = read_timeseries_csv(path, self.feature_names)
            raw[stay] = {"times": times, "values": values, "masks": masks}

        torch.save(
            {
                "version": FEATURE_CACHE_VERSION,
                "feature_names": self.feature_names,
                "stays": raw,
            },
            self.raw_cache_path,
        )
        print(f"Saved structured raw cache: {self.raw_cache_path}")
        return raw

    def compute_normalizer(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sums = torch.zeros(self.n_features, dtype=torch.float64)
        sq_sums = torch.zeros(self.n_features, dtype=torch.float64)
        counts = torch.zeros(self.n_features, dtype=torch.float64)

        for item in self.raw.values():
            values = item["values"].double()
            masks = item["masks"]
            if values.numel() == 0:
                continue
            masked = values * masks.double()
            sums += masked.sum(dim=0)
            sq_sums += (values.square() * masks.double()).sum(dim=0)
            counts += masks.double().sum(dim=0)

        counts = counts.clamp_min(1.0)
        mean = sums / counts
        var = (sq_sums / counts - mean.square()).clamp_min(1e-6)
        std = torch.sqrt(var)
        self.set_normalizer(mean.float(), std.float())
        return self.mean, self.std

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)
        self.prepared.clear()

    def _prepare_stay(self, stay: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if stay in self.prepared:
            return self.prepared[stay]
        if self.mean is None or self.std is None:
            raise RuntimeError("Call compute_normalizer() or set_normalizer() first.")

        item = self.raw[stay]
        times = item["times"].float()
        values = item["values"].float()
        masks = item["masks"].bool()
        n = times.shape[0]
        f = self.n_features
        if n == 0:
            feat = torch.empty(0, f * 3, dtype=torch.float32)
            self.prepared[stay] = (times, feat)
            return self.prepared[stay]

        norm_values = (values - self.mean.view(1, -1)) / self.std.view(1, -1)
        ffilled = torch.zeros(n, f, dtype=torch.float32)
        deltas = torch.ones(n, f, dtype=torch.float32)
        last_value = torch.zeros(f, dtype=torch.float32)
        last_time = torch.full((f,), float("nan"), dtype=torch.float32)

        for i in range(n):
            t = times[i]
            observed = masks[i]
            has_last = ~torch.isnan(last_time)
            current_delta = torch.full((f,), self.max_delta_hours, dtype=torch.float32)
            current_delta[has_last] = (t - last_time[has_last]).clamp(0, self.max_delta_hours)
            current_delta[observed] = 0.0
            deltas[i] = current_delta / self.max_delta_hours

            if observed.any():
                last_value[observed] = norm_values[i, observed]
                last_time[observed] = t
            ffilled[i] = last_value

        feat = torch.cat([ffilled, masks.float(), deltas], dim=-1)
        self.prepared[stay] = (times, feat)
        return self.prepared[stay]

    def make_sequence(
        self,
        stay: str,
        period_length: float,
        max_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        times, feat = self._prepare_stay(stay)
        out = torch.zeros(max_steps, self.input_dim, dtype=torch.float32)
        seq_mask = torch.zeros(max_steps, dtype=torch.bool)
        if times.numel() == 0:
            return out, seq_mask

        end = int(torch.searchsorted(times, torch.tensor(float(period_length)), right=True).item())
        if end <= 0:
            return out, seq_mask

        start = max(0, end - max_steps)
        chosen_times = times[start:end]
        chosen_feat = feat[start:end]
        n = chosen_feat.shape[0]
        age = ((float(period_length) - chosen_times) / 24.0).clamp(0, self.max_age_days)
        age = (age / self.max_age_days).unsqueeze(-1)
        seq = torch.cat([chosen_feat, age], dim=-1)
        out[:n] = seq
        seq_mask[:n] = True
        return out, seq_mask


class StructuredLOSDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[LOSSample],
        store: StayTimeSeriesStore,
        max_steps: int = 128,
        indices: Optional[Sequence[int]] = None,
    ):
        self.samples = samples
        self.store = store
        self.max_steps = max_steps
        self.indices = list(indices) if indices is not None else list(range(len(samples)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        raw_idx = self.indices[idx]
        sample = self.samples[raw_idx]
        x, seq_mask = self.store.make_sequence(sample.stay, sample.period_length, self.max_steps)
        return {
            "x": x,
            "seq_mask": seq_mask,
            "period_length": torch.tensor(sample.period_length, dtype=torch.float32),
            "y_true": torch.tensor(sample.y_true, dtype=torch.float32),
            "y_bin": torch.tensor(int(hours_to_bin([sample.y_true])[0]), dtype=torch.long),
            "stay": sample.stay,
            "raw_idx": raw_idx,
        }


def labels_for_indices(samples: Sequence[LOSSample], indices: Sequence[int]) -> np.ndarray:
    return np.asarray([samples[i].y_true for i in indices], dtype=np.float32)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_hours: np.ndarray,
    y_bin_pred: np.ndarray,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred_hours = np.asarray(y_pred_hours, dtype=np.float32)
    y_bin_true = hours_to_bin(y_true)
    y_bin_reg = hours_to_bin(y_pred_hours)
    return {
        "kappa_cls": sk_metrics.cohen_kappa_score(y_bin_true, y_bin_pred, weights="linear"),
        "kappa_reg": sk_metrics.cohen_kappa_score(y_bin_true, y_bin_reg, weights="linear"),
        "accuracy": sk_metrics.accuracy_score(y_bin_true, y_bin_pred) * 100.0,
        "mad": sk_metrics.mean_absolute_error(y_true, y_pred_hours),
        "mse": sk_metrics.mean_squared_error(y_true, y_pred_hours),
        "mape": np.mean(np.abs((y_true - y_pred_hours) / (y_true + 0.1))) * 100.0,
    }


def to_python_metrics(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def write_prediction_csv(path: str, rows: Iterable[Sequence]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stay", "predicted_bin", "true_bin", "pred_hours", "true_hours"])
        writer.writerows(rows)
