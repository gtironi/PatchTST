import argparse
import json
import os
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _is_sensor_col(col: str) -> bool:
    # Expected: ABack_x, ANeck_y, GBack_z, ...
    if not isinstance(col, str):
        return False
    if not (col.startswith("A") or col.startswith("G")):
        return False
    if not ("Back" in col or "Neck" in col):
        return False
    return col.endswith(("_x", "_y", "_z"))


def _clean_label(x):
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "<undefined>":
        return None
    if s == "Synchronization" or s == "Extra_Synchronization":
        return None
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_csv",
        type=str,
        default="PatchTST_self_supervised/dataset/DogMoveData.csv",
        help="Raw DogMoveData CSV (can be very large).",
    )
    p.add_argument(
        "--out_pretrain",
        type=str,
        default="PatchTST_self_supervised/dataset/dogmove_pretrain.parquet",
        help="Output Parquet for pretraining (date + sensors only, all rows).",
    )
    p.add_argument(
        "--out_classify",
        type=str,
        default="PatchTST_self_supervised/dataset/dogmove_classification.parquet",
        help="Output Parquet for classification (DogID/TestNum/date + sensors + Behavior_1, filtered).",
    )
    p.add_argument("--time_col", type=str, default="t_sec", help="Time column in seconds.")
    p.add_argument("--label_col", type=str, default="Behavior_1", help="Primary behavior label column.")
    p.add_argument("--chunksize", type=int, default=1_000_000, help="CSV chunk size.")
    p.add_argument(
        "--pretrain_only_labeled",
        action="store_true",
        help="If set, pretrain output will also be filtered to only rows with valid labels (same filter as classification). Useful for quick tests.",
    )
    p.add_argument(
        "--percentage",
        type=float,
        default=1.0,
        help="Write only this fraction of the rows (approx). Use e.g. 0.1 for a quick test run.",
    )
    p.add_argument(
        "--write_summary_json",
        type=str,
        default="PatchTST_self_supervised/dataset/dogmove_classification_labels.json",
        help="Write label counts summary to JSON (from filtered classification rows).",
    )
    args = p.parse_args()
    if not (0 < args.percentage <= 1.0):
        raise ValueError("--percentage must be in (0, 1].")

    os.makedirs(os.path.dirname(args.out_pretrain) or ".", exist_ok=True)

    # First chunk to infer columns
    head = pd.read_csv(args.input_csv, nrows=1)
    cols = list(head.columns)

    if args.time_col not in cols:
        raise ValueError(f"Missing time column {args.time_col!r}. Found: {cols}")
    if args.label_col not in cols:
        raise ValueError(f"Missing label column {args.label_col!r}. Found: {cols}")

    sensor_cols = [c for c in cols if _is_sensor_col(c)]
    if len(sensor_cols) == 0:
        raise ValueError(
            "No sensor columns found. Expected columns like ABack_x, ANeck_y, GBack_z, ..."
        )

    group_cols = [c for c in ["DogID", "TestNum"] if c in cols]
    pretrain_cols = [args.time_col] + sensor_cols
    classify_cols = group_cols + [args.time_col] + sensor_cols + [args.label_col]

    # Prepare outputs (overwrite)
    for outp in [args.out_pretrain, args.out_classify]:
        if os.path.exists(outp):
            os.remove(outp)

    label_counts = Counter()

    pretrain_writer = None
    classify_writer = None

    reader = pd.read_csv(args.input_csv, chunksize=args.chunksize)
    for chunk in reader:
        if args.percentage < 1.0:
            pre_chunk = chunk.sample(frac=args.percentage, random_state=0)
        else:
            pre_chunk = chunk

        # Pretrain: date + sensors, all rows (or only labeled if requested)
        if args.pretrain_only_labeled:
            tmp = pre_chunk[classify_cols].copy()
            tmp[args.label_col] = tmp[args.label_col].map(_clean_label)
            pre_chunk = tmp[tmp[args.label_col].notna()]

        pre = pre_chunk[pretrain_cols].copy()
        pre.insert(0, "date", pd.to_datetime(pre[args.time_col], unit="s", origin="unix", errors="coerce"))
        pre = pre.drop(columns=[args.time_col])
        if len(pre) > 0:
            pre_table = pa.Table.from_pandas(pre, preserve_index=False)
            if pretrain_writer is None:
                pretrain_writer = pq.ParquetWriter(args.out_pretrain, pre_table.schema, compression="snappy")
            pretrain_writer.write_table(pre_table)

        # Classification: keep only labeled rows, drop Synchronization/<undefined>/empty
        cls = chunk[classify_cols].copy()
        cls[args.label_col] = cls[args.label_col].map(_clean_label)
        cls = cls[cls[args.label_col].notna()]
        if args.percentage < 1.0 and len(cls) > 0:
            cls = cls.sample(frac=args.percentage, random_state=0)
        if len(cls) > 0:
            label_counts.update(cls[args.label_col].tolist())
            cls.insert(len(group_cols), "date", pd.to_datetime(cls[args.time_col], unit="s", origin="unix", errors="coerce"))
            cls = cls.drop(columns=[args.time_col])
            cls_table = pa.Table.from_pandas(cls, preserve_index=False)
            if classify_writer is None:
                classify_writer = pq.ParquetWriter(args.out_classify, cls_table.schema, compression="snappy")
            classify_writer.write_table(cls_table)

    if pretrain_writer is not None:
        pretrain_writer.close()
    if classify_writer is not None:
        classify_writer.close()

    # Summary
    if args.write_summary_json:
        summary = {
            "input_csv": args.input_csv,
            "out_pretrain": args.out_pretrain,
            "out_classify": args.out_classify,
            "time_col": args.time_col,
            "label_col": args.label_col,
            "sensor_cols": sensor_cols,
            "group_cols": group_cols,
            "label_counts": dict(label_counts.most_common()),
        }
        os.makedirs(os.path.dirname(args.write_summary_json) or ".", exist_ok=True)
        with open(args.write_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"- pretrain: {args.out_pretrain}")
    print(f"- classify: {args.out_classify}")
    if args.write_summary_json:
        print(f"- label summary: {args.write_summary_json}")


if __name__ == "__main__":
    main()

