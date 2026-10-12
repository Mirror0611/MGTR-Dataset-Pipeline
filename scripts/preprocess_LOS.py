from __future__ import annotations

import os
import csv
import math
from multiprocessing import Pool, cpu_count

import torch

from tqdm import tqdm
from transformers import AutoTokenizer


# MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_NAME = "D:\科研\电子病历关系与时间建模\models\Bio_ClinicalBERT"
MAX_LEN = 512
MAX_ROWS = 64
MAX_FEATURES_PER_ROW = 10

DATA_DIR = "./dataset/length-of-stay"

OUTPUT_DIR = "./processed_los"

NUM_WORKERS = min(16, cpu_count())

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = None


def init_tokenizer():
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


def parse_listfile(listfile_csv):

    out = []

    with open(listfile_csv, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            out.append(
                (
                    row["stay"],
                    float(row["period_length"]),
                    float(row["y_true"]),
                )
            )

    return out


def read_timeseries_rows(timeseries_csv, time_bound):

    with open(timeseries_csv, "r", encoding="utf-8") as f:

        reader = csv.reader(f)

        header = next(reader)

        rows = []

        for r in reader:

            if not r:
                continue

            try:
                t = float(r[0])
            except:
                continue

            if t > time_bound:
                break

            rows.append(r)

    return header, rows


def serialize_timeseries_to_text(
    header,
    rows,
    period_length,
):

    if not rows:
        return (
            f"ICU time series within first "
            f"{period_length:.1f} hours."
        )

    # Keep the newest observations when tokenizer truncation hits MAX_LEN.
    kept = list(reversed(rows[-MAX_ROWS:]))

    parts = [
        f"ICU time series within first "
            f"{period_length:.1f} hours. Most recent rows first."
    ]

    for r in kept:

        t = r[0]

        feats = []

        for i in range(1, min(len(header), len(r))):

            v = (r[i] or "").strip()

            if v == "" or v.lower() == "nan":
                continue

            feats.append(f"{header[i]}={v}")

            if len(feats) >= MAX_FEATURES_PER_ROW:
                break

        if feats:

            parts.append(
                f"Hour {t}: " +
                "; ".join(feats) + "."
            )

    return " ".join(parts)


def process_one(args):

    idx, split_dir, save_dir, sample = args

    stay, period_length, y_true = sample

    ts_path = os.path.join(split_dir, stay)

    header, rows = read_timeseries_rows(
        ts_path,
        period_length
    )

    text = serialize_timeseries_to_text(
        header,
        rows,
        period_length
    )

    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    data = {
        "input_ids":
            enc["input_ids"].squeeze(0),

        "attention_mask":
            enc["attention_mask"].squeeze(0),

        "y_log":
            torch.tensor(
                math.log1p(max(y_true, 0.0)),
                dtype=torch.float32,
            ),

        "y_true":
            torch.tensor(
                y_true,
                dtype=torch.float32,
            ),

        "stay":
            stay,

        "period_length":
            period_length,
    }

    save_path = os.path.join(
        save_dir,
        f"{idx}.pt"
    )

    torch.save(data, save_path)

    return save_path


def process_split(split_name):

    print(
        f"\n========== "
        f"Processing {split_name} "
        f"=========="
    )

    split_dir = os.path.join(
        DATA_DIR,
        split_name
    )

    save_dir = os.path.join(
        OUTPUT_DIR,
        split_name
    )

    os.makedirs(save_dir, exist_ok=True)

    listfile = os.path.join(
        split_dir,
        "listfile.csv"
    )

    samples = parse_listfile(listfile)

    tasks = []

    for idx, sample in enumerate(samples):

        tasks.append(
            (
                idx,
                split_dir,
                save_dir,
                sample
            )
        )

    meta = []

    with Pool(
        processes=NUM_WORKERS,
        initializer=init_tokenizer
    ) as pool:

        for save_path in tqdm(
            pool.imap_unordered(
                process_one,
                tasks,
                chunksize=32,
            ),
            total=len(tasks),
            desc=split_name,
        ):

            meta.append(save_path)

    torch.save(
        meta,
        os.path.join(save_dir, "index.pt")
    )

    print(
        f"{split_name} done. "
        f"total={len(meta)}"
    )


def main():

    print(
        f"CPU workers = {NUM_WORKERS}"
    )

    process_split("train")

    process_split("test")

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
