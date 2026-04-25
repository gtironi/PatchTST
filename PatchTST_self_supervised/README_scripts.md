## PatchTST self-supervised scripts (quick use)

### `patchtst_pretrain.py`
**What it does**: self-supervised *masked patch* pretraining. It masks a fraction of input patches (`--mask_ratio`) and trains PatchTST with `head_type='pretrain'` using MSE loss.

**Outputs**:
- Saves best checkpoint to `saved_models/<dset_pretrain>/masked_patchtst/<model_type>/<save_pretrained_model>.pth`
- Writes losses CSV: `.../<save_pretrained_model>_losses.csv`

**How to run**:

```bash
python PatchTST_self_supervised/patchtst_pretrain.py \
  --dset_pretrain etth1 --context_points 512 --target_points 96 \
  --patch_len 12 --stride 12 --mask_ratio 0.4 --n_epochs_pretrain 10
```

**Key flags**: `--dset_pretrain`, `--context_points`, `--target_points`, `--patch_len`, `--stride`, `--mask_ratio`, `--n_epochs_pretrain`, `--model_type`, `--pretrained_model_id`.

---

### `patchtst_finetune.py`
**What it does**: fine-tunes a pretrained PatchTST for forecasting (`head_type='prediction'`) or runs *linear probe* (train only the prediction head).

**Modes**:
- **Finetune**: `--is_finetune 1`
- **Linear probe**: `--is_linear_probe 1`
- **Test only**: neither flag set (it will test a weight path it constructs)

**Requires**: `--pretrained_model` pointing to the pretrained weights path (the code passes it into `transfer_weights(...)`; commonly this is a `.pth` path from pretraining).

**Outputs**:
- Saves best checkpoint to `saved_models/<dset_finetune>/masked_patchtst/<model_type>/<save_finetuned_model>.pth`
- Writes losses CSV: `.../<save_finetuned_model>_losses.csv`
- Writes test scores CSV: `.../<save_finetuned_model>_acc.csv` (mse, mae)

**How to run (finetune)**:

```bash
python PatchTST_self_supervised/patchtst_finetune.py \
  --is_finetune 1 --dset_finetune etth1 \
  --context_points 512 --target_points 96 --patch_len 12 --stride 12 \
  --pretrained_model saved_models/etth1/masked_patchtst/based_model/<PRETRAIN_NAME>.pth
```

**How to run (linear probe)**: swap `--is_finetune 1` for `--is_linear_probe 1`.

---

### `patchtst_supervised.py`
**What it does**: fully supervised forecasting training from scratch (no pretraining). Uses `head_type='prediction'`, MSE loss, reports `mse` metric, and saves best model by `valid_loss`.

**Outputs**:
- Saves best checkpoint to `saved_models/<dset>/patchtst_supervised/<model_type>/<save_model_name>.pth`

**How to run (train)**:

```bash
python PatchTST_self_supervised/patchtst_supervised.py \
  --is_train 1 --dset etth1 --context_points 336 --target_points 96 \
  --patch_len 32 --stride 16 --n_epochs 20
```

**How to run (test)**:

```bash
python PatchTST_self_supervised/patchtst_supervised.py --is_train 0 \
  --dset etth1 --context_points 336 --target_points 96 --patch_len 32 --stride 16
```

---

### Dataset / CSV expectations (from `src/data/`)
**CSV structure the code expects**
- **First column must be a timestamp column** (default name: `date`). The loaders assume it’s column index 0.
- **All other columns must be numeric time series variables** (float/int). Missing values should be handled before training.
- For `features='M'` / `features='MS'`: it uses **all columns except the first** (`df_raw.columns[1:]`).
- For `features='S'`: it uses **only one column** named by `target` (default in ETT datasets is `OT`).

**How windows are built**
- Each sample returns `seq_x` of length `context_points` and `seq_y` covering the forecast horizon `target_points` (internally `size=[context_points, 0, target_points]`).
- Optional time features: set `--use_time_features 1` to also return calendar features derived from the timestamp.

**Adapting to a new dataset (do you need code changes?)**
- If you can **reuse an existing `--dset` name** and place your CSV in the hardcoded `root_path` used in `datautils.get_dls()`, you can avoid edits.
- Otherwise, **yes—edit code**: add your dataset name to `DSETS` and add a new `elif` in `PatchTST_self_supervised/datautils.py` pointing to your `root_path` + `data_path`, typically using `Dataset_Custom`.
- Minimal requirement to work with `Dataset_Custom`: a CSV with timestamp in column 0 (prefer name `date`) + numeric columns; pass `--features M` (multivariate) or `--features S` plus make sure `target` matches a column name.

**Is the data different for pretrain vs downstream?**
- **Same input format and dataloader** (`get_dls(args)`) for all three scripts.
- The difference is the **training objective/callbacks**:
  - Pretrain: masks patches (`PatchMaskCB`) and learns to reconstruct (self-supervised).
  - Finetune / Supervised: no masking (`PatchCB`), predicts future horizon (forecasting).

