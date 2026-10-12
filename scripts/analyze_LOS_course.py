from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path


def _early_set_cuda_visible_devices() -> None:
    gpu = None
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--gpu" and i + 1 < len(argv):
            gpu = argv[i + 1]
            break
        if arg.startswith("--gpu="):
            gpu = arg.split("=", 1)[1]
            break
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu).strip()
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("LOS_GPU", "0").strip() or "0"
    os.environ.setdefault("OMP_NUM_THREADS", "1")


_early_set_cuda_visible_devices()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from los_structured_utils import StayTimeSeriesStore, hours_to_bin, parse_listfile, safe_torch_load
from train_LOS_fusion import (
    ClinicalBertFusionAdapter,
    FusionLOSDataset,
    build_or_load_bert_feature_cache,
    load_text_split,
)
from train_LOS_task2_prior import FusionTask2PriorLOSDataset, autocast_context
from train_LOS_task2_time_aligned_prior import (
    TASK2_DRUG_CLASSES,
    build_or_load_time_aligned_prior,
    phenotype_dim_for_agg,
)
from train_LOS_V1 import ClinicalBertStructuredAdapter


PHENOTYPE_LABELS = ["TOXICITY", "REBOUND", "TARGET_ACHIEVED", "REFRACTORY"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate long-term disease-course analysis from a LOS checkpoint.")
    parser.add_argument("--checkpoint", default="./los_best_task2_time_aligned_prior_fusion.pt")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--output-dir", default="./los_course_analysis")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--risk-thresholds-hours", default="168,336")
    parser.add_argument("--num-case-plots", type=int, default=30)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional quick-run limit for debugging.")
    parser.add_argument(
        "--predictions-csv",
        default="./los_predictions_task2_time_aligned_prior_fusion.csv",
        help=(
            "Optional existing main-model prediction CSV. If present, analysis reuses it instead of "
            "running the LOS model forward. The compact training CSV format "
            "stay,predicted_bin,true_bin,pred_hours,true_hours is supported."
        ),
    )
    parser.add_argument(
        "--force-model-predict",
        action="store_true",
        help="Ignore --predictions-csv and run model prediction from the checkpoint.",
    )
    parser.add_argument("--rebuild-task2-prior-cache", action="store_true")
    parser.add_argument("--rebuild-bert-cache", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--baseline-checkpoint",
        default="./los_best_fusion.pt",
        help=(
            "Optional checkpoint from a no-Task2-prior LOS model, e.g. ./los_best_fusion.pt. "
            "When provided, a baseline prediction curve is added to CSVs and case plots."
        ),
    )
    parser.add_argument(
        "--baseline-predictions-csv",
        default="los_predictions_fusion.csv",
        help=(
            "Optional precomputed no-prior predictions CSV. It should contain stay, period_length, "
            "and a prediction column such as pred_remaining_hours_reg or pred_hours. If period_length "
            "is absent, row order must match the current split exactly."
        ),
    )
    parser.add_argument("--baseline-label", default="No Task2 prior")
    parser.add_argument("--baseline-batch-size", type=int, default=None)
    parser.add_argument(
        "--clear-old-case-plots",
        action="store_true",
        help="Remove existing PNGs under output-dir/case_plots before drawing the newly selected cases.",
    )
    return parser


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def make_loader(dataset, batch_size: int, num_workers: int, pin_memory: bool):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)

@torch.no_grad()
def prediction_rows(model, loader, device, use_bert_cache: bool, prior_dim: int, task2_agg: str):
    model.eval()
    rows = []
    phenotype_dim = phenotype_dim_for_agg(task2_agg)
    if task2_agg == "mean_max":
        phenotype_cols = [f"task2_mean_{x.lower()}" for x in PHENOTYPE_LABELS] + [
            f"task2_max_{x.lower()}" for x in PHENOTYPE_LABELS
        ]
    else:
        phenotype_cols = [f"task2_{x.lower()}" for x in PHENOTYPE_LABELS]

    for batch in tqdm(loader, desc="Predicting"):
        if use_bert_cache:
            bert_input = batch["bert_features"].to(device, non_blocking=True)
            attention_mask = None
        else:
            bert_input = batch["input_ids"].long().to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].long().to(device, non_blocking=True)
        x = batch["x"].to(device, non_blocking=True)
        seq_mask = batch["seq_mask"].to(device, non_blocking=True)
        period_length = batch["period_length"].to(device, non_blocking=True)
        with autocast_context(device):
            reg_pred, cls_logits = model(bert_input, attention_mask, x, seq_mask, period_length)

        pred_hours = torch.expm1(reg_pred).clamp_min(0).float().cpu().numpy()
        cls_prob = torch.softmax(cls_logits, dim=-1).float().cpu().numpy()
        pred_bins = cls_prob.argmax(axis=1)
        true_hours = batch["y_true"].numpy()
        periods = batch["period_length"].numpy()
        true_bins = hours_to_bin(true_hours)
        stays = list(batch["stay"])
        priors = batch.get("task2_prior", None)
        if priors is None:
            prior_np = np.zeros((len(stays), prior_dim), dtype=np.float32)
        else:
            prior_np = priors.numpy()

        for i, stay in enumerate(stays):
            row = {
                "stay": stay,
                "period_length": float(periods[i]),
                "true_remaining_hours": float(true_hours[i]),
                "true_total_icu_hours_est": float(periods[i] + true_hours[i]),
                "pred_remaining_hours_reg": float(pred_hours[i]),
                "pred_bin_cls": int(pred_bins[i]),
                "true_bin": int(true_bins[i]),
                "abs_error_hours": float(abs(pred_hours[i] - true_hours[i])),
            }
            for b in range(cls_prob.shape[1]):
                row[f"cls_prob_bin{b}"] = float(cls_prob[i, b])
            for j, col in enumerate(phenotype_cols):
                if j < prior_np.shape[1]:
                    row[col] = float(prior_np[i, j])
            drug_offset = phenotype_dim
            for j, drug in enumerate(TASK2_DRUG_CLASSES):
                idx = drug_offset + j
                if idx < prior_np.shape[1]:
                    row[f"drug_{drug.lower().replace(' ', '_').replace('-', '_')}"] = float(prior_np[i, idx])
            if prior_np.shape[1] > 0:
                row["task2_missing_indicator"] = float(prior_np[i, -1])
            rows.append(row)
    return pd.DataFrame(rows)

@torch.no_grad()
def simple_prediction_rows(model, loader, device, use_bert_cache: bool) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="Predicting baseline"):
        if use_bert_cache:
            bert_input = batch["bert_features"].to(device, non_blocking=True)
            attention_mask = None
        else:
            bert_input = batch["input_ids"].long().to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].long().to(device, non_blocking=True)
        x = batch["x"].to(device, non_blocking=True)
        seq_mask = batch["seq_mask"].to(device, non_blocking=True)
        period_length = batch["period_length"].to(device, non_blocking=True)
        with autocast_context(device):
            reg_pred, cls_logits = model(bert_input, attention_mask, x, seq_mask, period_length)

        pred_hours = torch.expm1(reg_pred).clamp_min(0).float().cpu().numpy()
        pred_bins = torch.softmax(cls_logits, dim=-1).argmax(dim=1).float().cpu().numpy()
        periods = batch["period_length"].numpy()
        stays = list(batch["stay"])
        for i, stay in enumerate(stays):
            rows.append(
                {
                    "stay": stay,
                    "period_length": float(periods[i]),
                    "pred_remaining_hours_baseline": float(pred_hours[i]),
                    "pred_bin_baseline": int(pred_bins[i]),
                }
            )
    return pd.DataFrame(rows)


def phenotype_column_names(task2_agg: str) -> list[str]:
    if task2_agg == "mean_max":
        return [f"task2_mean_{x.lower()}" for x in PHENOTYPE_LABELS] + [
            f"task2_max_{x.lower()}" for x in PHENOTYPE_LABELS
        ]
    return [f"task2_{x.lower()}" for x in PHENOTYPE_LABELS]


def enrich_prediction_frame(
    pred_df: pd.DataFrame,
    samples,
    prior: np.ndarray,
    prior_dim: int,
    task2_agg: str,
    require_order_if_no_period: bool = True,
) -> pd.DataFrame:
    out = pred_df.copy()
    if "stay" not in out.columns:
        raise ValueError(f"Prediction CSV must include stay column; got columns={list(out.columns)}")

    if "pred_remaining_hours_reg" not in out.columns:
        if "pred_hours" in out.columns:
            out["pred_remaining_hours_reg"] = out["pred_hours"].astype(float)
        elif "pred_remaining_hours" in out.columns:
            out["pred_remaining_hours_reg"] = out["pred_remaining_hours"].astype(float)
        else:
            raise ValueError(
                "Prediction CSV must contain pred_remaining_hours_reg, pred_remaining_hours, or pred_hours."
            )
    if "true_remaining_hours" not in out.columns:
        if "true_hours" in out.columns:
            out["true_remaining_hours"] = out["true_hours"].astype(float)
        else:
            out["true_remaining_hours"] = [float(s.y_true) for s in samples[: len(out)]]
    if "pred_bin_cls" not in out.columns:
        if "predicted_bin" in out.columns:
            out["pred_bin_cls"] = out["predicted_bin"].astype(int)
        elif "pred_bin" in out.columns:
            out["pred_bin_cls"] = out["pred_bin"].astype(int)
        else:
            out["pred_bin_cls"] = hours_to_bin(out["pred_remaining_hours_reg"].to_numpy(dtype=float)).astype(int)
    if "true_bin" not in out.columns:
        out["true_bin"] = hours_to_bin(out["true_remaining_hours"].to_numpy(dtype=float)).astype(int)

    sample_stays = np.asarray([s.stay for s in samples], dtype=object)
    sample_periods = np.asarray([float(s.period_length) for s in samples], dtype=np.float32)
    sample_true = np.asarray([float(s.y_true) for s in samples], dtype=np.float32)

    if "period_length" not in out.columns:
        if require_order_if_no_period:
            if len(out) != len(samples):
                raise ValueError(
                    f"Prediction CSV has no period_length and length {len(out)} != samples {len(samples)}. "
                    "Use an analysis CSV with period_length or do not set --max-rows."
                )
            if not (out["stay"].astype(str).to_numpy() == sample_stays.astype(str)).all():
                raise ValueError(
                    "Prediction CSV has no period_length and stay order does not match LOS listfile order."
                )
            out["period_length"] = sample_periods
            if "true_hours" not in pred_df.columns and "true_remaining_hours" not in pred_df.columns:
                out["true_remaining_hours"] = sample_true
        else:
            raise ValueError("Prediction CSV must include period_length.")

    if len(out) == len(samples) and (out["stay"].astype(str).to_numpy() == sample_stays.astype(str)).all():
        prior_for_rows = prior[: len(out)]
    else:
        keys = {}
        for i, sample in enumerate(samples):
            keys[(str(sample.stay), round(float(sample.period_length), 5))] = i
        row_indices = []
        for _, row in out.iterrows():
            idx = keys.get((str(row["stay"]), round(float(row["period_length"]), 5)))
            row_indices.append(idx)
        if any(i is None for i in row_indices):
            missing = sum(i is None for i in row_indices)
            raise ValueError(f"Could not align {missing} prediction rows to LOS samples by stay+period_length.")
        prior_for_rows = prior[np.asarray(row_indices, dtype=int)]

    out["period_length"] = out["period_length"].astype(float)
    out["true_remaining_hours"] = out["true_remaining_hours"].astype(float)
    out["true_total_icu_hours_est"] = out["period_length"] + out["true_remaining_hours"]
    out["pred_remaining_hours_reg"] = out["pred_remaining_hours_reg"].astype(float)
    out["abs_error_hours"] = (out["pred_remaining_hours_reg"] - out["true_remaining_hours"]).abs()

    phenotype_dim = phenotype_dim_for_agg(task2_agg)
    for j, col in enumerate(phenotype_column_names(task2_agg)):
        if col not in out.columns and j < prior_for_rows.shape[1]:
            out[col] = prior_for_rows[:, j].astype(float)
    drug_offset = phenotype_dim
    for j, drug in enumerate(TASK2_DRUG_CLASSES):
        idx = drug_offset + j
        col = f"drug_{drug.lower().replace(' ', '_').replace('-', '_')}"
        if col not in out.columns and idx < prior_for_rows.shape[1]:
            out[col] = prior_for_rows[:, idx].astype(float)
    if "task2_missing_indicator" not in out.columns and prior_dim > 0 and prior_for_rows.shape[1] > 0:
        out["task2_missing_indicator"] = prior_for_rows[:, -1].astype(float)
    return out


def load_main_predictions_csv(path: str, samples, prior: np.ndarray, prior_dim: int, task2_agg: str) -> pd.DataFrame:
    print(f"Loading existing LOS predictions: {path}")
    pred_df = pd.read_csv(path)
    return enrich_prediction_frame(pred_df, samples, prior, prior_dim, task2_agg)


def _checkpoint_args(ckpt: dict) -> argparse.Namespace:
    return argparse.Namespace(**ckpt["args"])


def _use_bert_cache_from_args(args: argparse.Namespace) -> bool:
    return not getattr(args, "no_bert_cache", False) and getattr(args, "unfreeze_last_n_layers", 0) == 0


def _ensure_batch_size(args: argparse.Namespace, override: int | None, use_bert_cache: bool) -> None:
    if override is not None:
        args.batch_size = override
    elif getattr(args, "batch_size", None) is None:
        args.batch_size = 256 if use_bert_cache else 32


def build_no_prior_model(
    ckpt: dict,
    args: argparse.Namespace,
    structured_input_dim: int,
    use_bert_cache: bool,
    bert_feature_dim: int,
):
    if hasattr(args, "struct_encoder"):
        return ClinicalBertFusionAdapter(
            model_name=args.model_name,
            structured_input_dim=structured_input_dim,
            use_cached_bert=use_bert_cache,
            bert_feature_dim=bert_feature_dim,
            struct_encoder=args.struct_encoder,
            struct_hidden_dim=args.struct_hidden_dim,
            struct_num_layers=args.struct_num_layers,
            temporal_conv_layers=args.temporal_conv_layers,
            conv_kernel_size=args.conv_kernel_size,
            pooling=args.pooling,
            fusion_mode=args.fusion_mode,
            fusion_hidden_dim=args.fusion_hidden_dim,
            text_proj_dim=args.text_proj_dim,
            adapter_bottleneck=args.adapter_bottleneck,
            dropout=args.dropout,
            unfreeze_last_n_layers=args.unfreeze_last_n_layers,
            gradient_checkpointing=args.gradient_checkpointing,
        )

    return ClinicalBertStructuredAdapter(
        model_name=args.model_name,
        structured_input_dim=structured_input_dim,
        use_cached_bert=use_bert_cache,
        bert_feature_dim=bert_feature_dim,
        struct_hidden_dim=args.struct_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        text_proj_dim=args.text_proj_dim,
        adapter_bottleneck=args.adapter_bottleneck,
        dropout=args.dropout,
        unfreeze_last_n_layers=args.unfreeze_last_n_layers,
        gradient_checkpointing=args.gradient_checkpointing,
    )


def predict_baseline_checkpoint(
    checkpoint_path: str,
    split_name: str,
    device,
    pin_memory: bool,
    cli,
) -> pd.DataFrame:
    print(f"Loading no-prior baseline checkpoint: {checkpoint_path}")
    ckpt = safe_torch_load(checkpoint_path)
    args = _checkpoint_args(ckpt)
    if getattr(args, "temporal_conv_layers", 0) < 0:
        args.temporal_conv_layers = 1 if getattr(args, "struct_encoder", "").startswith("tcn_") else 0
    use_bert_cache = _use_bert_cache_from_args(args)
    _ensure_batch_size(args, cli.baseline_batch_size or cli.batch_size, use_bert_cache)
    args.num_workers = cli.num_workers
    if hasattr(args, "rebuild_bert_cache"):
        args.rebuild_bert_cache = bool(cli.rebuild_bert_cache)
    if hasattr(args, "rebuild_cache"):
        args.rebuild_cache = bool(cli.rebuild_cache)

    samples = parse_listfile(os.path.join(args.data_dir, split_name, "listfile.csv"))
    if cli.max_rows and cli.max_rows > 0:
        samples = samples[: cli.max_rows]

    text_data = load_text_split(args.processed_dir, split_name)
    bert_features = None
    bert_feature_dim = 768
    if use_bert_cache:
        bert_features = build_or_load_bert_feature_cache(
            args.model_name,
            text_data,
            split_name,
            args.bert_cache_dir,
            args.precompute_bert_batch_size,
            device,
            getattr(args, "rebuild_bert_cache", False),
        )
        if cli.max_rows and cli.max_rows > 0:
            bert_features = bert_features[: cli.max_rows]
        bert_feature_dim = int(bert_features.shape[1])

    store = StayTimeSeriesStore(
        args.data_dir,
        split_name,
        samples,
        cache_dir=args.cache_dir,
        feature_names=ckpt.get("feature_names"),
        rebuild_cache=getattr(args, "rebuild_cache", False),
    )
    store.set_normalizer(ckpt["mean"], ckpt["std"])
    dataset = FusionLOSDataset(samples, store, text_data, args.max_steps, None, bert_features)
    loader = make_loader(dataset, args.batch_size, args.num_workers, pin_memory)
    model = build_no_prior_model(ckpt, args, store.input_dim, use_bert_cache, bert_feature_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    return simple_prediction_rows(model, loader, device, use_bert_cache)


def load_baseline_predictions_csv(path: str, df: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(path)
    pred_candidates = [
        "pred_remaining_hours_baseline",
        "pred_remaining_hours_reg",
        "pred_remaining_hours",
        "pred_hours",
        "baseline_pred_remaining_hours",
    ]
    pred_col = next((c for c in pred_candidates if c in base.columns), None)
    if pred_col is None:
        raise ValueError(
            f"Baseline CSV must include one of {pred_candidates}; got columns={list(base.columns)}"
        )
    if "stay" not in base.columns:
        raise ValueError("Baseline CSV must include a stay column.")

    out = base.copy()
    out["pred_remaining_hours_baseline"] = out[pred_col].astype(float)
    keep_cols = ["stay", "pred_remaining_hours_baseline"]
    if "period_length" in out.columns:
        keep_cols.insert(1, "period_length")
        return out[keep_cols]

    if len(out) != len(df) or not (out["stay"].astype(str).to_numpy() == df["stay"].astype(str).to_numpy()).all():
        raise ValueError(
            "Baseline CSV has no period_length, so it can only be used when row order and stays "
            "match the current prediction table exactly."
        )
    aligned = df[["stay", "period_length"]].copy()
    aligned["pred_remaining_hours_baseline"] = out["pred_remaining_hours_baseline"].to_numpy()
    return aligned


def merge_baseline_predictions(df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    left = out.copy()
    right = baseline_df.copy()
    left["_period_key"] = left["period_length"].astype(float).round(5)
    right["_period_key"] = right["period_length"].astype(float).round(5)
    right = right.rename(columns={"stay": "_baseline_stay"})
    merged = left.merge(
        right[["_baseline_stay", "_period_key", "pred_remaining_hours_baseline"]],
        left_on=["stay", "_period_key"],
        right_on=["_baseline_stay", "_period_key"],
        how="left",
    )
    merged = merged.drop(columns=["_period_key", "_baseline_stay"])
    merged["delta_pred_task2_minus_baseline_hours"] = (
        merged["pred_remaining_hours_reg"] - merged["pred_remaining_hours_baseline"]
    )
    matched = int(merged["pred_remaining_hours_baseline"].notna().sum())
    print(f"Merged baseline predictions: matched={matched}/{len(merged)}")
    return merged


def add_risk_columns(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    out = df.copy()
    for th in thresholds:
        label = f"gt_{int(th)}h"
        out[f"true_risk_{label}"] = (out["true_remaining_hours"] > th).astype(int)
        out[f"pred_risk_reg_{label}"] = (out["pred_remaining_hours_reg"] > th).astype(int)
    return out


def add_phenotype_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_cols = [f"task2_{x.lower()}" for x in PHENOTYPE_LABELS]
    if not all(c in out.columns for c in raw_cols):
        return out
    raw = out[raw_cols].fillna(0.0).to_numpy(dtype=float)
    denom = raw.sum(axis=1, keepdims=True)
    norm = np.divide(raw, np.maximum(denom, 1e-8), out=np.zeros_like(raw), where=denom > 0)
    for j, label in enumerate(PHENOTYPE_LABELS):
        out[f"task2_norm_{label.lower()}"] = norm[:, j]
    idx = raw.argmax(axis=1)
    has_prior = raw.sum(axis=1) > 0
    out["task2_dominant_phenotype"] = [
        PHENOTYPE_LABELS[int(i)] if has_prior[k] else "NONE" for k, i in enumerate(idx)
    ]
    out["task2_raw_sum"] = raw.sum(axis=1)
    return out


def risk_summary(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for th in thresholds:
        label = f"gt_{int(th)}h"
        y = df[f"true_risk_{label}"].to_numpy()
        p = df[f"pred_risk_reg_{label}"].to_numpy()
        tp = int(((y == 1) & (p == 1)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        acc = (tp + tn) / max(len(y), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        rows.append(
            {
                "threshold_hours": th,
                "threshold_days": th / 24.0,
                "support_positive": int(y.sum()),
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return pd.DataFrame(rows)


def phenotype_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c
        for c in df.columns
        if c.startswith("task2_")
        and c
        not in {
            "task2_missing_indicator",
            "task2_dominant_phenotype",
            "task2_raw_sum",
        }
    ]
    rows = []
    for col in cols:
        vals = df[col].to_numpy(dtype=float)
        pred = df["pred_remaining_hours_reg"].to_numpy(dtype=float)
        true = df["true_remaining_hours"].to_numpy(dtype=float)
        if np.nanstd(vals) < 1e-8:
            corr_pred = np.nan
            corr_true = np.nan
        else:
            corr_pred = float(np.corrcoef(vals, pred)[0, 1])
            corr_true = float(np.corrcoef(vals, true)[0, 1])
        high = df[vals >= np.nanquantile(vals, 0.75)] if len(vals) else df.iloc[[]]
        low = df[vals <= np.nanquantile(vals, 0.25)] if len(vals) else df.iloc[[]]
        rows.append(
            {
                "feature": col,
                "mean": float(np.nanmean(vals)) if len(vals) else np.nan,
                "corr_with_pred_remaining": corr_pred,
                "corr_with_true_remaining": corr_true,
                "pred_remaining_high_q75": float(high["pred_remaining_hours_reg"].mean()) if len(high) else np.nan,
                "pred_remaining_low_q25": float(low["pred_remaining_hours_reg"].mean()) if len(low) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def trajectory_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stay, g in df.sort_values(["stay", "period_length"]).groupby("stay"):
        if len(g) < 2:
            continue
        dominant_counts = {}
        if "task2_dominant_phenotype" in g.columns:
            dominant_counts = g["task2_dominant_phenotype"].value_counts().to_dict()
        rows.append(
            {
                "stay": stay,
                "n_points": len(g),
                "first_period": float(g["period_length"].iloc[0]),
                "last_period": float(g["period_length"].iloc[-1]),
                "first_pred_remaining": float(g["pred_remaining_hours_reg"].iloc[0]),
                "last_pred_remaining": float(g["pred_remaining_hours_reg"].iloc[-1]),
                "delta_pred_remaining": float(
                    g["pred_remaining_hours_reg"].iloc[-1] - g["pred_remaining_hours_reg"].iloc[0]
                ),
                "mean_abs_error": float(g["abs_error_hours"].mean()),
                "max_task2_toxicity": float(g.get("task2_toxicity", pd.Series([np.nan])).max()),
                "max_task2_rebound": float(g.get("task2_rebound", pd.Series([np.nan])).max()),
                "max_task2_target_achieved": float(g.get("task2_target_achieved", pd.Series([np.nan])).max()),
                "max_task2_refractory": float(g.get("task2_refractory", pd.Series([np.nan])).max()),
                "dominant_toxicity_points": int(dominant_counts.get("TOXICITY", 0)),
                "dominant_rebound_points": int(dominant_counts.get("REBOUND", 0)),
                "dominant_target_points": int(dominant_counts.get("TARGET_ACHIEVED", 0)),
                "dominant_refractory_points": int(dominant_counts.get("REFRACTORY", 0)),
                "dominant_none_points": int(dominant_counts.get("NONE", 0)),
            }
        )
    return pd.DataFrame(rows)


def plot_cases(
    df: pd.DataFrame,
    out_dir: Path,
    n_cases: int,
    baseline_label: str = "No Task2 prior",
    clear_old_plots: bool = False,
) -> None:
    plot_dir = out_dir / "case_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if clear_old_plots:
        for old_png in plot_dir.glob("*.png"):
            old_png.unlink()
    traj = trajectory_summary(df)
    if traj.empty:
        return
    case_stats = []
    has_baseline = "pred_remaining_hours_baseline" in df.columns
    for stay, g in df.groupby("stay", sort=False):
        if len(g) < 2:
            continue
        stat = {"stay": stay, "n_points": len(g)}
        if "task2_raw_sum" in g.columns:
            raw = g["task2_raw_sum"].fillna(0.0).to_numpy(dtype=float)
            stat["has_task2_prior"] = float(np.nanmax(raw) > 0.0)
            stat["task2_prior_changes"] = float(np.sum(np.abs(np.diff(raw)) > 1e-6))
            stat["max_task2_raw_sum"] = float(np.nanmax(raw))
        else:
            stat["has_task2_prior"] = 0.0
            stat["task2_prior_changes"] = 0.0
            stat["max_task2_raw_sum"] = 0.0
        if has_baseline:
            diff = (g["pred_remaining_hours_reg"] - g["pred_remaining_hours_baseline"]).abs()
            stat["mean_baseline_delta_days"] = float(diff.mean() / 24.0)
            stat["max_baseline_delta_days"] = float(diff.max() / 24.0)
        else:
            stat["mean_baseline_delta_days"] = 0.0
            stat["max_baseline_delta_days"] = 0.0
        case_stats.append(stat)
    candidates = pd.DataFrame(case_stats)
    if candidates.empty:
        return
    candidates["plot_score"] = (
        np.log1p(candidates["n_points"])
        + 4.0 * candidates["has_task2_prior"]
        + 0.25 * candidates["task2_prior_changes"]
        + 0.35 * candidates["mean_baseline_delta_days"]
        + 0.10 * candidates["max_baseline_delta_days"]
    )
    candidates = candidates.sort_values("plot_score", ascending=False).head(min(n_cases, len(candidates)))
    for stay in candidates["stay"]:
        g = df[df["stay"] == stay].sort_values("period_length")
        fig, ax1 = plt.subplots(figsize=(9, 5))
        x_days = g["period_length"] / 24.0
        ax1.plot(x_days, g["true_remaining_hours"] / 24.0, label="True remaining LOS", color="#2f5d8c")
        ax1.plot(x_days, g["pred_remaining_hours_reg"] / 24.0, label="Predicted remaining LOS", color="#c44e52")
        if "pred_remaining_hours_baseline" in g.columns and g["pred_remaining_hours_baseline"].notna().any():
            ax1.plot(
                x_days,
                g["pred_remaining_hours_baseline"] / 24.0,
                label=baseline_label,
                color="#666666",
                linestyle=":",
                linewidth=2.0,
            )
        ax1.set_xlabel("ICU elapsed time (days)")
        ax1.set_ylabel("Remaining LOS (days)")
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        for col, color in [
            ("task2_norm_toxicity", "#8c564b"),
            ("task2_norm_rebound", "#9467bd"),
            ("task2_norm_target_achieved", "#2ca02c"),
            ("task2_norm_refractory", "#ff7f0e"),
        ]:
            if col in g.columns:
                ax2.plot(
                    x_days,
                    g[col],
                    linestyle="--",
                    drawstyle="steps-post",
                    alpha=0.75,
                    label=col.replace("task2_norm_", ""),
                    color=color,
                )
        ax2.set_ylabel("Task2 phenotype prior (row-normalized for display)")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
        fig.suptitle(stay)
        fig.tight_layout()
        safe_name = stay.replace(".csv", "").replace("/", "_").replace("\\", "_")
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=180)
        plt.close(fig)


def write_analysis_notes(out_dir: Path, task2_agg: str, baseline_label: str, has_baseline: bool) -> None:
    baseline_note = ""
    if has_baseline:
        baseline_note = f"""
## Baseline comparison

Case plots include a `{baseline_label}` curve when baseline predictions were provided.
The CSV column `delta_pred_task2_minus_baseline_hours` is:

`Task2-prior model predicted remaining hours - no-prior baseline predicted remaining hours`.

Positive values mean Task2 priors increased the LOS estimate relative to the no-prior model.
Negative values mean Task2 priors decreased the LOS estimate.
"""
    notes = f"""# LOS Course Analysis Notes

## Task2 phenotype columns

Raw `task2_toxicity`, `task2_rebound`, `task2_target_achieved`, and `task2_refractory` are model input prior features.
When `task2_agg={task2_agg}`, these values may be aggregated over multiple visible drug-lab trajectories for the same ICU stay and time point.

If `task2_agg=max`, each phenotype is the maximum probability observed among visible drug-lab trajectories. Therefore the four raw columns are not a single softmax distribution and can be high at the same time.

For display, the script also writes row-normalized columns:

- `task2_norm_toxicity`
- `task2_norm_rebound`
- `task2_norm_target_achieved`
- `task2_norm_refractory`

Case plots use the normalized columns, while the CSV preserves the raw model-input priors.
{baseline_note}
"""
    (out_dir / "README_analysis.md").write_text(notes, encoding="utf-8")


def main() -> None:
    cli = build_arg_parser().parse_args()
    ckpt = safe_torch_load(cli.checkpoint)
    args = argparse.Namespace(**ckpt["args"])
    args.rebuild_task2_prior_cache = bool(cli.rebuild_task2_prior_cache)
    args.rebuild_bert_cache = bool(cli.rebuild_bert_cache)
    args.rebuild_cache = bool(cli.rebuild_cache)
    if cli.batch_size is not None:
        args.batch_size = cli.batch_size
    args.num_workers = cli.num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    use_bert_cache = not getattr(args, "no_bert_cache", False) and getattr(args, "unfreeze_last_n_layers", 0) == 0
    if args.batch_size is None:
        args.batch_size = 256 if use_bert_cache else 32

    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device = {device}")
    print(f"Checkpoint = {cli.checkpoint}")
    print(f"Output dir = {out_dir}")

    train_samples = parse_listfile(os.path.join(args.data_dir, "train", "listfile.csv"))
    test_samples = parse_listfile(os.path.join(args.data_dir, "test", "listfile.csv"))
    train_prior, test_prior, prior_dim = build_or_load_time_aligned_prior(args, train_samples, test_samples, device)

    if cli.split == "test":
        samples = test_samples
        split_name = "test"
        prior = test_prior
    else:
        samples = train_samples
        split_name = "train"
        prior = train_prior

    if cli.max_rows and cli.max_rows > 0:
        samples = samples[: cli.max_rows]
        prior = prior[: cli.max_rows]

    can_reuse_predictions = (
        cli.predictions_csv
        and not cli.force_model_predict
        and os.path.exists(cli.predictions_csv)
    )
    if can_reuse_predictions:
        df = load_main_predictions_csv(cli.predictions_csv, samples, prior, prior_dim, args.task2_agg)
    else:
        if cli.predictions_csv and not cli.force_model_predict:
            print(f"Prediction CSV not found, falling back to checkpoint inference: {cli.predictions_csv}")
        text_data = load_text_split(args.processed_dir, split_name)
        bert_features = None
        bert_feature_dim = 768
        if use_bert_cache:
            bert_features = build_or_load_bert_feature_cache(
                args.model_name,
                text_data,
                split_name,
                args.bert_cache_dir,
                args.precompute_bert_batch_size,
                device,
                args.rebuild_bert_cache,
            )
            if cli.max_rows and cli.max_rows > 0:
                bert_features = bert_features[: cli.max_rows]
            bert_feature_dim = int(bert_features.shape[1])

        store = StayTimeSeriesStore(
            args.data_dir,
            split_name,
            samples,
            cache_dir=args.cache_dir,
            feature_names=ckpt.get("feature_names"),
            rebuild_cache=args.rebuild_cache,
        )
        store.set_normalizer(ckpt["mean"], ckpt["std"])
        dataset = FusionTask2PriorLOSDataset(
            samples,
            store,
            text_data,
            args.max_steps,
            None,
            bert_features,
            prior,
        )
        loader = make_loader(dataset, args.batch_size, args.num_workers, pin_memory)

        model = ClinicalBertFusionAdapter(
            model_name=args.model_name,
            structured_input_dim=store.input_dim + prior_dim,
            use_cached_bert=use_bert_cache,
            bert_feature_dim=bert_feature_dim,
            struct_encoder=args.struct_encoder,
            struct_hidden_dim=args.struct_hidden_dim,
            struct_num_layers=args.struct_num_layers,
            temporal_conv_layers=args.temporal_conv_layers,
            conv_kernel_size=args.conv_kernel_size,
            pooling=args.pooling,
            fusion_mode=args.fusion_mode,
            fusion_hidden_dim=args.fusion_hidden_dim,
            text_proj_dim=args.text_proj_dim,
            adapter_bottleneck=args.adapter_bottleneck,
            dropout=args.dropout,
            unfreeze_last_n_layers=args.unfreeze_last_n_layers,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])

        df = prediction_rows(model, loader, device, use_bert_cache, prior_dim, args.task2_agg)
    thresholds = parse_float_list(cli.risk_thresholds_hours)
    df = add_phenotype_display_columns(df)
    has_baseline = False
    if cli.baseline_predictions_csv and os.path.exists(cli.baseline_predictions_csv):
        baseline_df = load_baseline_predictions_csv(cli.baseline_predictions_csv, df)
        df = merge_baseline_predictions(df, baseline_df)
        has_baseline = True
    elif cli.baseline_checkpoint and os.path.exists(cli.baseline_checkpoint):
        baseline_df = predict_baseline_checkpoint(cli.baseline_checkpoint, split_name, device, pin_memory, cli)
        df = merge_baseline_predictions(df, baseline_df)
        has_baseline = True
    elif cli.baseline_predictions_csv:
        print(f"Baseline prediction CSV not found, skipping baseline curve: {cli.baseline_predictions_csv}")
    elif cli.baseline_checkpoint:
        print(f"Baseline checkpoint not found, skipping baseline curve: {cli.baseline_checkpoint}")
    df = add_risk_columns(df, thresholds)
    pred_path = out_dir / f"{split_name}_course_predictions.csv"
    df.to_csv(pred_path, index=False)

    risk = risk_summary(df, thresholds)
    risk.to_csv(out_dir / f"{split_name}_risk_summary.csv", index=False)
    phen = phenotype_summary(df)
    phen.to_csv(out_dir / f"{split_name}_phenotype_explanation.csv", index=False)
    traj = trajectory_summary(df)
    traj.to_csv(out_dir / f"{split_name}_trajectory_summary.csv", index=False)
    plot_cases(df, out_dir, cli.num_case_plots, cli.baseline_label, cli.clear_old_case_plots)
    write_analysis_notes(out_dir, args.task2_agg, cli.baseline_label, has_baseline)

    print(f"Saved predictions: {pred_path}")
    print(f"Saved risk summary: {out_dir / f'{split_name}_risk_summary.csv'}")
    print(f"Saved phenotype explanation: {out_dir / f'{split_name}_phenotype_explanation.csv'}")
    print(f"Saved trajectory summary: {out_dir / f'{split_name}_trajectory_summary.csv'}")
    print(f"Saved case plots under: {out_dir / 'case_plots'}")
    print(f"Saved analysis notes: {out_dir / 'README_analysis.md'}")


if __name__ == "__main__":
    main()
