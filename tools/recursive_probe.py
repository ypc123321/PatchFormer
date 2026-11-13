# -*- coding: utf-8 -*-
"""Closed-loop recursive extrapolation probe.

The benchmark script evaluates one-step-ahead predictions on every real
history window (teacher forcing).  This probe implements the *other* protocol
described in the paper: feed the prediction back into the input window and
iterate until the capacity crosses the EOL threshold.

It intentionally uses only the capacity channel (the model input is
``encoder_cont[:, :, :-1]``), exactly as in training.  The output is a JSON
record plus a figure that can be included in the paper.

Usage (from ``02_代码``):
    python tools/recursive_probe.py
    python tools/recursive_probe.py --battery CS2_35 --sp 300 --steps 1200
"""
from __future__ import print_function

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from CALCEDataPreProcess import BatteryDataProcess  # noqa: E402
from ModelsModify.PatchFormer import PatchFormerNetModel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--battery', default='CS2_35')
    p.add_argument('--sp', type=int, default=300)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--rated', type=float, default=1.1)
    p.add_argument('--threshold_ratio', type=float, default=0.7,
                   help='EOL threshold as a fraction of rated capacity (0.7 -> 0.77 Ah)')
    p.add_argument('--weights', default=None,
                   help='checkpoint path; defaults to weights/PatchFormer/<battery>/best.ckpt')
    p.add_argument('--data_dir', default='data/CALCE data/')
    p.add_argument('--output_json', default='results/recursive_probe.json')
    p.add_argument('--output_figure', default='figures/fig7_recursive.png')
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = args.weights or os.path.join(
        'weights', 'PatchFormer', args.battery, 'best.ckpt')
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    data = np.load(os.path.join(args.data_dir, 'CALCE_Data.npy'),
                   allow_pickle=True).item()
    if args.battery not in data:
        raise KeyError('battery {} not in {}'.format(args.battery, args.data_dir))

    # Global min/max used by CALCEDataPreProcess for the Capacity feature.
    train_cap = np.concatenate([
        df['Capacity'].values for name, df in data.items() if name != args.battery
    ]) / args.rated
    cap_min = float(train_cap.min())
    cap_max = float(train_cap.max())

    ns = SimpleNamespace(seq_len=args.seq_len, pred_len=1,
                         Rated_Capacity=args.rated,
                         start_point_list=[args.sp],
                         data_dir=args.data_dir, test_name=args.battery)
    _, _, df_all = BatteryDataProcess(data, args.battery, args.sp, ns)

    window = df_all.loc[
        (df_all['Cycle'] >= args.sp - args.seq_len + 1) &
        (df_all['Cycle'] <= args.sp), 'target'
    ].values.astype(np.float64)
    if len(window) != args.seq_len:
        raise RuntimeError('expected {} history points, got {}'.format(
            args.seq_len, len(window)))

    model = PatchFormerNetModel.load_from_checkpoint(ckpt, map_location='cpu')
    model.eval()

    threshold = args.rated * args.threshold_ratio
    preds_ah = []
    preds_target = []
    with torch.no_grad():
        for _ in range(args.steps):
            center = float(window.mean())
            # EncoderNormalizer uses the window standard deviation.  A floor
            # avoids 0/0 once the recursion reaches an exact fixed point.
            scale = max(float(window.std()), 1e-6)
            mm = (window - cap_min) / (cap_max - cap_min)
            x = torch.tensor(mm, dtype=torch.float32).reshape(
                1, args.seq_len, 1)
            raw = float(model.network(x).reshape(-1)[0])
            pred_t = raw * scale + center
            preds_target.append(pred_t)
            preds_ah.append(pred_t * args.rated)
            window = np.append(window[1:], pred_t)

    preds_ah = np.asarray(preds_ah)
    preds_target = np.asarray(preds_target)
    below = np.where(preds_ah <= threshold)[0]
    first_below = int(below[0]) + 1 if len(below) else None
    max_idx = int(np.argmax(preds_ah))
    result = {
        'model': 'PatchFormer',
        'battery': args.battery,
        'dataset': 'CALCE',
        'sp': args.sp,
        'seq_len': args.seq_len,
        'steps': args.steps,
        'rated_capacity_ah': args.rated,
        'eol_threshold_ah': threshold,
        'train_capacity_min': cap_min,
        'train_capacity_max': cap_max,
        'start_cycle': args.sp + 1,
        'end_cycle': args.sp + args.steps,
        'initial_prediction_ah': float(preds_ah[0]),
        'max_prediction_ah': float(preds_ah[max_idx]),
        'max_prediction_cycle': args.sp + max_idx + 1,
        'final_prediction_ah': float(preds_ah[-1]),
        'min_prediction_ah': float(preds_ah.min()),
        'threshold_crossed': first_below is not None,
        'first_below_cycle': first_below,
        'note': 'Closed-loop recursion with capacity-only input; prediction is fed back into the window.',
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Figure: measured curve, recursive prediction, EOL threshold.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
        'legend.fontsize': 9, 'figure.dpi': 120, 'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    cap = data[args.battery]['Capacity'].values * args.rated
    cycles = np.arange(1, len(cap) + 1)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(cycles, cap, color='#1f77b4', lw=1.1, label='Measurement')
    pred_cycles = np.arange(args.sp + 1, args.sp + 1 + len(preds_ah))
    ax.plot(pred_cycles, preds_ah, color='#d62728', lw=1.4,
            label='Closed-loop recursion')
    ax.axhline(threshold, ls='--', lw=0.9, color='k',
               label='EOL threshold ({:.2f} Ah)'.format(threshold))
    ax.axvline(args.sp, ls=':', lw=0.8, color='gray')
    ax.annotate('fixed point {:.3f} Ah'.format(result['final_prediction_ah']),
                xy=(pred_cycles[-1], result['final_prediction_ah']),
                xytext=(pred_cycles[-1] - 260, result['final_prediction_ah'] + 0.025),
                fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Capacity (Ah)')
    ax.set_title('Closed-loop recursive extrapolation ({} , SP={})'.format(
        args.battery, args.sp))
    ax.set_xlim(0, min(len(cap), args.sp + len(preds_ah)) + 20)
    ax.set_ylim(min(cap.min(), threshold) - 0.03, cap.max() + 0.04)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc='upper right')
    os.makedirs(os.path.dirname(os.path.abspath(args.output_figure)), exist_ok=True)
    fig.savefig(args.output_figure)
    plt.close(fig)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('saved', args.output_json)
    print('saved', args.output_figure)


if __name__ == '__main__':
    main()