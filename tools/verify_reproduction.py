# -*- coding: utf-8 -*-
"""Reproduce the reported metrics from the released checkpoints.

For every checkpoint in weights/, this script rebuilds the exact test split used by
RUL_Prediction_benchmark.py and compares the recomputed MAE / RMSE / R2 / RUL AE with
results/summary_metrics_agg.csv.

Run from 02_代码:
    python tools/verify_reproduction.py
    python tools/verify_reproduction.py --models PatchFormer,PatchTST
"""
import argparse
import csv
import importlib
import os
import sys
import warnings
from types import SimpleNamespace

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer
from sklearn.metrics import r2_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from CALCEDataPreProcess import BatteryDataProcess as CalceProcess
from TJUDataPreProcess import BatteryDataProcess as TjuProcess

MODEL_CLASSES = {
    "PatchFormer": ("ModelsModify.PatchFormer", "PatchFormerNetModel"),
    "PatchFormer_no_DPAN": ("ModelsModify.PatchFormer_no_DPAN", "PatchFormer_no_DPANNetModel"),
    "PatchFormer_no_inter_PA": ("ModelsModify.PatchFormer_no_inter_PA", "PatchFormer_no_inter_PANetModel"),
    "PatchFormer_no_intra_PA": ("ModelsModify.PatchFormer_no_intra_PA", "PatchFormer_no_intra_PANetModel"),
    "PatchTST": ("ModelsModify.PatchTST", "PatchTSTNetModel"),
    "iTransformer": ("ModelsModify.iTransformer", "iTransformerModel"),
    "ModernTCN": ("ModelsModify.ModernTCN", "ModernTCNModel"),
}

# (model, battery, dataset, [start points])
JOBS = [
    ("PatchFormer", "CS2_35", "CALCE", [300, 400, 500]),
    ("PatchFormer", "CS2_37", "CALCE", [300, 400, 500]),
    ("PatchFormer", "CY25_1", "TJU", [200, 300, 400]),
    ("PatchFormer_no_DPAN", "CS2_35", "CALCE", [300, 400, 500]),
    ("PatchFormer_no_inter_PA", "CS2_35", "CALCE", [300, 400, 500]),
    ("PatchFormer_no_intra_PA", "CS2_35", "CALCE", [300, 400, 500]),
    ("PatchTST", "CS2_35", "CALCE", [300, 400, 500]),
    ("PatchTST", "CS2_37", "CALCE", [300, 400, 500]),
    ("PatchTST", "CY25_1", "TJU", [200, 300, 400]),
    ("iTransformer", "CS2_35", "CALCE", [300, 400, 500]),
    ("iTransformer", "CS2_37", "CALCE", [300, 400, 500]),
    ("iTransformer", "CY25_1", "TJU", [200, 300, 400]),
    ("ModernTCN", "CS2_35", "CALCE", [300, 400, 500]),
    ("ModernTCN", "CS2_37", "CALCE", [300, 400, 500]),
    ("ModernTCN", "CY25_1", "TJU", [200, 300, 400]),
]


def rul_value_error(y_test, y_predict, threshold):
    """Same implementation as the benchmark script, kept for an apples-to-apples check."""
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test) - 1):
        if y_test[i] <= threshold >= y_test[i + 1]:
            true_re = i - 1
            break
    for i in range(len(y_predict) - 1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    rul_real = true_re + 1
    rul_pred = pred_re + 1
    ae_error = abs(true_re - pred_re)
    re_score = min(abs(true_re - pred_re) / true_re, 1.0)
    return rul_real, rul_pred, ae_error, re_score


def load_data(dataset):
    if dataset == "TJU":
        path = os.path.join("data", "TJU data", "Dataset_3_NCM_NCA_battery_1C.npy")
        return np.load(path, allow_pickle=True).item(), TjuProcess, 2.5
    path = os.path.join("data", "CALCE data", "CALCE_Data.npy")
    return np.load(path, allow_pickle=True).item(), CalceProcess, 1.1


def build_loader(df_test):
    ds = TimeSeriesDataSet(
        df_test,
        time_idx="time_idx",
        target="target",
        group_ids=["group_id"],
        min_encoder_length=64,
        max_encoder_length=64,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_known_reals=["Capacity"],
        time_varying_unknown_reals=["target"],
        target_normalizer=EncoderNormalizer(),
        add_encoder_length=False,
    )
    return ds.to_dataloader(train=False, batch_size=256, shuffle=False, num_workers=0, drop_last=False)


def load_model(model_name, checkpoint):
    module_name, class_name = MODEL_CLASSES[model_name]
    cls = getattr(importlib.import_module(module_name), class_name)
    model = cls.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="", help="comma separated model names; default: all")
    parser.add_argument("--report", default="results/verification_report.csv")
    args = parser.parse_args()

    selected = {x.strip() for x in args.models.split(",") if x.strip()}
    jobs = [j for j in JOBS if not selected or j[0] in selected]
    if not jobs:
        raise SystemExit("no matching jobs")

    reported = {}
    with open("results/summary_metrics_agg.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            reported[(row["model"], row["test_name"], int(row["sp"]))] = row

    data_cache = {}
    rows = []
    worst = 0.0
    for model_name, battery, dataset, sps in jobs:
        ckpt = os.path.join("weights", model_name, battery, "best.ckpt")
        if not os.path.exists(ckpt):
            print("[SKIP] missing", ckpt)
            continue
        if dataset not in data_cache:
            data_cache[dataset] = load_data(dataset)
        data, process, rated = data_cache[dataset]
        model = load_model(model_name, ckpt)
        for sp in sps:
            ns = SimpleNamespace(seq_len=64, pred_len=1, Rated_Capacity=rated,
                                 start_point_list=[sp], data_dir="", test_name=battery)
            _, df_test, df_all = process(data, battery, sp, ns)
            loader = build_loader(df_test)
            with torch.no_grad():
                pred = model.predict(loader, batch_size=256).detach().cpu().numpy().reshape(-1)
            actual = df_all.loc[df_all["Cycle"] >= sp, "target"].values * rated
            pred = pred * rated
            n = min(len(actual), len(pred))
            actual, pred = actual[:n], pred[:n]
            mae = float(np.mean(np.abs(actual - pred)))
            rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
            r2 = float(r2_score(actual, pred))
            rul_real, rul_pred, ae, re = rul_value_error(actual, pred, rated * 0.7)
            ref = reported.get((model_name, battery, sp), {})
            diffs = []
            for key, value in (("MAE", mae), ("RMSE", rmse), ("r2", r2)):
                if ref.get(key) not in (None, ""):
                    diffs.append(abs(value - float(ref[key])))
            if ref.get("AE") not in (None, ""):
                diffs.append(abs(ae - float(ref["AE"])))
            diff = max(diffs) if diffs else float("nan")
            worst = max(worst, diff)
            rows.append({
                "model": model_name, "test_name": battery, "sp": sp,
                "MAE_reported": ref.get("MAE", ""), "MAE_recomputed": round(mae, 6),
                "RMSE_reported": ref.get("RMSE", ""), "RMSE_recomputed": round(rmse, 6),
                "R2_reported": ref.get("r2", ""), "R2_recomputed": round(r2, 6),
                "AE_reported": ref.get("AE", ""), "AE_recomputed": int(ae),
                "max_abs_diff": round(diff, 6),
                "status": "PASS" if diff <= 1e-3 else "CHECK",
            })
            print("{} {} SP{} MAE {:.6f} (csv {}) AE {} (csv {}) {}".format(
                model_name, battery, sp, mae, ref.get("MAE", "?"), ae, ref.get("AE", "?"),
                rows[-1]["status"]))

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nrows:", len(rows), "worst_abs_diff:", round(worst, 6))
    print("report:", args.report)
    if worst > 1e-3:
        raise SystemExit("verification failed: at least one metric differs by > 1e-3")


if __name__ == "__main__":
    main()