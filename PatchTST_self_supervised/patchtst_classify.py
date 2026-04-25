import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.basics import set_device
from src.callback.patch_mask import PatchCB
from src.callback.tracking import SaveModelCB
from src.callback.transforms import RevInCB
from src.learner import Learner, transfer_weights
from src.models.patchTST import PatchTST


class MeanPoolClassificationHead(nn.Module):
    """
    Input:  x [bs, nvars, d_model, num_patch]
    Output: y [bs, n_classes]
    """

    def __init__(self, n_vars: int, d_model: int, n_classes: int, head_dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(n_vars * d_model, n_classes)

    def forward(self, x):
        x = x.mean(dim=-1)  # [bs, nvars, d_model]
        x = self.flatten(x)  # [bs, nvars*d_model]
        x = self.dropout(x)
        return self.linear(x)


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # y_true: [bs], y_pred: [bs, n_classes]
    preds = torch.argmax(y_pred, dim=-1)
    return (preds == y_true).float().mean()


def reduce_label(labels: Sequence[str], how: str) -> str:
    if len(labels) == 0:
        raise ValueError("Empty label window.")
    if how == "last":
        return labels[-1]
    if how == "center":
        return labels[len(labels) // 2]
    if how == "majority":
        # fast majority for short windows
        vals, counts = np.unique(np.asarray(labels, dtype=object), return_counts=True)
        return str(vals[int(np.argmax(counts))])
    raise ValueError(f"Unknown label_reduce={how!r} (expected majority|last|center).")


@dataclass
class SplitData:
    X: np.ndarray  # [N, seq_len, n_vars]
    y_str: List[str]


def build_windows(
    df: pd.DataFrame,
    sensor_cols: List[str],
    label_col: str,
    group_cols: List[str],
    seq_len: int,
    window_stride: int,
    label_reduce: str,
    max_windows: Optional[int] = None,
) -> SplitData:
    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    if group_cols:
        grouped = df.groupby(group_cols, sort=False)
        groups = (g for _, g in grouped)
    else:
        groups = (df,)

    for gdf in groups:
        gdf = gdf.reset_index(drop=True)
        x = gdf[sensor_cols].to_numpy(dtype=np.float32, copy=True)
        y = gdf[label_col].astype(str).to_numpy(dtype=object, copy=False)

        T = len(gdf)
        if T < seq_len:
            continue

        for start in range(0, T - seq_len + 1, window_stride):
            end = start + seq_len
            X_list.append(x[start:end])
            y_list.append(reduce_label(y[start:end], label_reduce))
            if max_windows is not None and len(X_list) >= max_windows:
                return SplitData(X=np.stack(X_list, axis=0), y_str=y_list)

    if len(X_list) == 0:
        raise ValueError("No windows created. Check seq_len/window_stride and filtering.")
    return SplitData(X=np.stack(X_list, axis=0), y_str=y_list)


def split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise ValueError("Invalid train/val ratios.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx = np.arange(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


class NumpyWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleDLS:
    def __init__(self, train, valid, test, n_vars: int):
        self.train = train
        self.valid = valid
        self.test = test
        self.vars = n_vars


def get_model(c_in: int, n_classes: int, args) -> PatchTST:
    num_patch = (max(args.context_points, args.patch_len) - args.patch_len) // args.stride + 1
    model = PatchTST(
        c_in=c_in,
        target_dim=n_classes,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act="relu",
        head_type="classification",
        res_attention=False,
    )
    return model


def main():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_path", type=str, default="dataset/dogmove_classification.csv")
    p.add_argument("--label_col", type=str, default="Behavior_1")
    p.add_argument("--group_cols", type=str, default="DogID,TestNum", help="Comma-separated group columns (optional).")
    p.add_argument("--context_points", type=int, default=512, help="Window length.")
    p.add_argument("--window_stride", type=int, default=1, help="Stride between windows.")
    p.add_argument("--label_reduce", type=str, default="majority", help="majority|last|center")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--max_windows", type=int, default=0, help="0 means no limit.")

    # Patch/model
    p.add_argument("--patch_len", type=int, default=12)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--revin", type=int, default=1)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--head_dropout", type=float, default=0.2)
    p.add_argument("--pooling_type", type=str, default="mean", help="mean|last")

    # Optimization
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze_epochs", type=int, default=10, help="Only used for end-to-end finetune.")
    p.add_argument("--is_linear_probe", type=int, default=0, help="If 1: freeze backbone and train only head.")

    # Pretrained + saving
    p.add_argument("--pretrained_model", type=str, required=True, help="Path to .pth from masked pretraining.")
    p.add_argument("--save_dir", type=str, default="saved_models/dogmove/classification/based_model/")
    p.add_argument("--run_name", type=str, default="patchtst_dogmove_classification")

    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_device()

    df = pd.read_csv(args.data_path)
    if args.label_col not in df.columns:
        raise ValueError(f"label_col {args.label_col!r} not in CSV columns.")

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip() and c.strip() in df.columns]

    # Sensors = all numeric columns excluding group/label/date
    excluded = set(group_cols + [args.label_col])
    if "date" in df.columns:
        excluded.add("date")
    sensor_cols = [c for c in df.columns if c not in excluded]

    # Ensure numeric
    df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=sensor_cols + [args.label_col]).reset_index(drop=True)

    max_windows = None if args.max_windows == 0 else int(args.max_windows)
    data = build_windows(
        df=df,
        sensor_cols=sensor_cols,
        label_col=args.label_col,
        group_cols=group_cols,
        seq_len=args.context_points,
        window_stride=args.window_stride,
        label_reduce=args.label_reduce,
        max_windows=max_windows,
    )

    n = data.X.shape[0]
    train_idx, val_idx, test_idx = split_indices(n, args.train_ratio, args.val_ratio)

    # label mapping from TRAIN only
    train_labels = [data.y_str[i] for i in train_idx]
    unique = sorted(set(train_labels))
    label2id: Dict[str, int] = {lab: i for i, lab in enumerate(unique)}
    id2label: Dict[str, str] = {str(i): lab for lab, i in label2id.items()}

    def encode_and_filter(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray([label2id.get(data.y_str[i], -1) for i in idx], dtype=np.int64)
        keep = y >= 0
        return data.X[idx][keep], y[keep]

    X_train, y_train = encode_and_filter(train_idx)
    X_val, y_val = encode_and_filter(val_idx)
    X_test, y_test = encode_and_filter(test_idx)

    n_vars = X_train.shape[-1]
    n_classes = len(label2id)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes in train split, got {n_classes}.")

    train_dl = DataLoader(
        NumpyWindowDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_dl = DataLoader(
        NumpyWindowDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_dl = DataLoader(
        NumpyWindowDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    dls = SimpleDLS(train=train_dl, valid=val_dl, test=test_dl, n_vars=n_vars)

    model = get_model(n_vars, n_classes, args)
    model = transfer_weights(args.pretrained_model, model, exclude_head=True, device="cpu")

    # Pooling choice (default: mean)
    if args.pooling_type == "mean":
        model.head = MeanPoolClassificationHead(n_vars, args.d_model, n_classes, args.head_dropout)
    elif args.pooling_type == "last":
        # keep default ClassificationHead behavior (last patch)
        pass
    else:
        raise ValueError("pooling_type must be mean|last")

    loss_func = nn.CrossEntropyLoss()

    cbs = []
    if args.revin:
        cbs.append(RevInCB(dls.vars, denorm=True))
    cbs += [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor="valid_loss", fname=args.run_name, path=args.save_dir),
    ]

    learn = Learner(dls, model, loss_func=loss_func, lr=args.lr, cbs=cbs, metrics=[accuracy])

    if args.is_linear_probe:
        learn.linear_probe(n_epochs=args.n_epochs, base_lr=args.lr)
    else:
        learn.fine_tune(n_epochs=args.n_epochs, base_lr=args.lr, freeze_epochs=args.freeze_epochs)

    # Save label mapping
    mapping_path = os.path.join(args.save_dir, f"{args.run_name}_label_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
                "sensor_cols": sensor_cols,
                "group_cols": group_cols,
                "label_col": args.label_col,
                "data_path": args.data_path,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Done.")
    print(f"- checkpoint dir: {args.save_dir}")
    print(f"- label mapping: {mapping_path}")


if __name__ == "__main__":
    main()

