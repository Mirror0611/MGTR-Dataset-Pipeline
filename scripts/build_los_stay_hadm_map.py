from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict


def norm_id(value) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def read_csv_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def collect_los_stays(los_dir: str, include_top_level_listfiles: bool = False) -> dict[str, set[str]]:
    stays: dict[str, set[str]] = defaultdict(set)
    candidate_paths = [
        ("train", os.path.join(los_dir, "train", "listfile.csv")),
        ("test", os.path.join(los_dir, "test", "listfile.csv")),
    ]
    if include_top_level_listfiles:
        candidate_paths.extend(
            [
                ("train", os.path.join(los_dir, "train_listfile.csv")),
                ("test", os.path.join(los_dir, "test_listfile.csv")),
                ("val", os.path.join(los_dir, "val_listfile.csv")),
            ]
        )
    for split, path in candidate_paths:
        if not os.path.exists(path):
            continue
        for row in read_csv_rows(path):
            stay = row.get("stay", "").strip()
            if stay:
                stays[stay].add(split)
    return stays


STAY_RE = re.compile(r"^(?P<subject>\d+)_episode(?P<episode>\d+)_timeseries\.csv$")


def parse_los_stay(stay: str) -> tuple[str, int] | None:
    match = STAY_RE.match(stay)
    if not match:
        return None
    return match.group("subject"), int(match.group("episode"))


def build_episode_mapping(root_dir: str, requested_stays: Iterable[str] | None = None) -> tuple[dict[str, dict], dict[str, int]]:
    mapping: dict[str, dict] = {}
    stats = defaultdict(int)

    if requested_stays is not None:
        by_subject: dict[str, dict[int, str]] = defaultdict(dict)
        for stay in requested_stays:
            parsed = parse_los_stay(stay)
            if parsed is None:
                stats["unparsed_los_stay"] += 1
                continue
            subject_id, episode_idx = parsed
            by_subject[subject_id][episode_idx] = stay

        for subject_id, episode_to_stay in sorted(by_subject.items()):
            subject_dir = ""
            root_partition = ""
            for partition in ("train", "test"):
                candidate = os.path.join(root_dir, partition, subject_id)
                if os.path.isdir(candidate):
                    subject_dir = candidate
                    root_partition = partition
                    break
            if not subject_dir:
                stats["missing_subject_dir"] += 1
                continue

            stays_path = os.path.join(subject_dir, "stays.csv")
            if not os.path.exists(stays_path):
                stats["missing_stays_csv"] += 1
                continue
            stay_rows = read_csv_rows(stays_path)

            for episode_idx, stay in episode_to_stay.items():
                row_idx = episode_idx - 1
                if row_idx < 0 or row_idx >= len(stay_rows):
                    stats["episode_index_out_of_range"] += 1
                    continue
                ts_path = os.path.join(subject_dir, f"episode{episode_idx}_timeseries.csv")
                if not os.path.exists(ts_path):
                    stats["episode_without_timeseries"] += 1
                    continue
                row = stay_rows[row_idx]
                mapping[stay] = {
                    "stay": stay,
                    "HADM_ID": norm_id(row.get("HADM_ID", "")),
                    "SUBJECT_ID": norm_id(row.get("SUBJECT_ID", subject_id)),
                    "ICUSTAY_ID": norm_id(row.get("ICUSTAY_ID", "")),
                    "episode": str(episode_idx),
                    "root_partition": root_partition,
                }
                stats["mapped_episodes"] += 1
        return mapping, dict(stats)

    for partition in ("train", "test"):
        partition_dir = os.path.join(root_dir, partition)
        if not os.path.isdir(partition_dir):
            stats[f"missing_{partition}_dir"] += 1
            continue

        for subject_id in sorted(os.listdir(partition_dir)):
            if not subject_id.isdigit():
                continue
            subject_dir = os.path.join(partition_dir, subject_id)
            if not os.path.isdir(subject_dir):
                continue
            stays_path = os.path.join(subject_dir, "stays.csv")
            if not os.path.exists(stays_path):
                stats["missing_stays_csv"] += 1
                continue

            for episode_idx, row in enumerate(read_csv_rows(stays_path), start=1):
                ts_name = f"episode{episode_idx}_timeseries.csv"
                ts_path = os.path.join(subject_dir, ts_name)
                if not os.path.exists(ts_path):
                    stats["episode_without_timeseries"] += 1
                    continue

                stay = f"{subject_id}_{ts_name}"
                if stay in mapping:
                    stats["duplicate_stay"] += 1
                mapping[stay] = {
                    "stay": stay,
                    "HADM_ID": norm_id(row.get("HADM_ID", "")),
                    "SUBJECT_ID": norm_id(row.get("SUBJECT_ID", subject_id)),
                    "ICUSTAY_ID": norm_id(row.get("ICUSTAY_ID", "")),
                    "episode": str(episode_idx),
                    "root_partition": partition,
                }
                stats["mapped_episodes"] += 1
    return mapping, dict(stats)


def load_task2_hadms(task2_csv: str | None) -> set[str]:
    if not task2_csv or not os.path.exists(task2_csv):
        return set()
    rows = read_csv_rows(task2_csv)
    return {norm_id(row.get("HADM_ID", "")) for row in rows if row.get("HADM_ID", "")}


def write_mapping(output_path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fieldnames = [
        "stay",
        "HADM_ID",
        "SUBJECT_ID",
        "ICUSTAY_ID",
        "episode",
        "root_partition",
        "los_splits",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build stay,HADM_ID mapping for MIMIC-III benchmark LOS files."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="mimic3-benchmarks data/root directory containing train/test subject folders.",
    )
    parser.add_argument(
        "--los-dir",
        default="./dataset/length-of-stay",
        help="LOS task directory containing train/test listfile.csv files.",
    )
    parser.add_argument("--task2-csv", default="./dataset/Task2_all.csv")
    parser.add_argument("--output", default="./stay_hadm_map.csv")
    parser.add_argument(
        "--include-top-level-listfiles",
        action="store_true",
        help="Also scan train_listfile.csv/test_listfile.csv/val_listfile.csv in los-dir.",
    )
    parser.add_argument(
        "--all-root-episodes",
        action="store_true",
        help="Write all root episodes instead of restricting to stays used by LOS listfiles.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root_dir = os.path.abspath(args.root_dir)
    los_dir = os.path.abspath(args.los_dir)
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"--root-dir not found: {root_dir}")
    if not args.all_root_episodes and not os.path.isdir(los_dir):
        raise NotADirectoryError(f"--los-dir not found: {los_dir}")

    los_stays = (
        collect_los_stays(los_dir, args.include_top_level_listfiles)
        if not args.all_root_episodes
        else {}
    )
    root_mapping, stats = build_episode_mapping(
        root_dir,
        None if args.all_root_episodes else los_stays.keys(),
    )
    if args.all_root_episodes:
        selected_stays = sorted(root_mapping)
    else:
        selected_stays = sorted(los_stays)

    rows = []
    missing = []
    for stay in selected_stays:
        item = root_mapping.get(stay)
        if item is None:
            missing.append(stay)
            continue
        out = dict(item)
        out["los_splits"] = ";".join(sorted(los_stays.get(stay, [])))
        rows.append(out)

    write_mapping(args.output, rows)

    task2_hadms = load_task2_hadms(args.task2_csv)
    task2_overlap = sum(1 for row in rows if row["HADM_ID"] in task2_hadms) if task2_hadms else 0
    unique_hadms = len({row["HADM_ID"] for row in rows if row["HADM_ID"]})

    print(f"Root episodes mapped: {len(root_mapping)}")
    print(f"LOS unique stays requested: {len(selected_stays)}")
    print(f"Rows written: {len(rows)} -> {args.output}")
    print(f"Missing LOS stays in root mapping: {len(missing)}")
    print(f"Unique HADM_ID in output: {unique_hadms}")
    if task2_hadms:
        print(
            f"Task2 HADM overlap: {task2_overlap}/{len(rows)} mapped stays "
            f"({100.0 * task2_overlap / max(len(rows), 1):.2f}%)"
        )
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")
    if missing[:5]:
        print("First missing stays:", ", ".join(missing[:5]))


if __name__ == "__main__":
    main()
