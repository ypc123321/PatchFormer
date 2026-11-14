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

    # Match the training pipeline exactly:
    #   1) Capacity (known real) is min-max scaled with the training batteries;
    #   2) the TimeSeriesDataSet then applies its StandardScaler to that column;
    #   3) target is normalized per window by EncoderNormalizer.
    train_cap = np.concatenate([
        df['Capacity'].values for name, df in data.items() if name != args.battery
    ]) / args.rated
    cap_min = float(train_cap.min())
    cap_max = float(train_cap.max())
    ns = SimpleNamespace(seq_len=args.seq_len, pred_len=1,
                         Rated_Capacity=args.rated,
                         start_point_list=[args.sp],
                         data_dir=args.data_dir, test_name=args.battery)
    df_train, _, df_all = BatteryDataProcess(data, args.battery, args.sp, ns)
    # BatteryDataProcess already min-max scales df_train's Capacity column;
    # its mean/std are exactly the StandardScaler statistics used for the
    # known-real input in training.
    cap_mean = float(df_train['Capacity'].mean())
    cap_std = float(df_train['Capacity'].std())
    # Target space is Capacity / rated (not min-max scaled).  The global
    # min/max below are used only by the diagnostic inverse control.
    target_min = float(df_train['target'].min())
    target_max = float(df_train['target'].max())

    init_window = df_all.loc[
        (df_all['Cycle'] >= args.sp - args.seq_len + 1) &
        (df_all['Cycle'] <= args.sp), 'target'
    ].values.astype(np.float64)
    if len(init_window) != args.seq_len:
        raise RuntimeError('expected {} history points, got {}'.format(
            args.seq_len, len(init_window)))

    model = PatchFormerNetModel.load_from_checkpoint(ckpt, map_location='cpu')
    model.eval()
    threshold = args.rated * args.threshold_ratio

    def simulate(mode, output_norm):
        """Run one closed-loop protocol.

        mode:
            'model'       - network prediction is fed back
            'persistence' - last observed capacity in the window is fed back
        output_norm (model only):
            'window'       - per-window EncoderNormalizer (training path)
            'global_minmax'- diagnostic global min-max inverse
        """
        window = init_window.copy()
        preds_ah, preds_t = [], []
        with torch.no_grad():
            for _ in range(args.steps):
                if mode == 'persistence':
                    pred_t = float(window[-1])
                else:
                    # Known-real input: min-max then StandardScaler, exactly
                    # as in the training/test datasets.
                    mm = (window - cap_min) / (cap_max - cap_min)
                    feat = (mm - cap_mean) / cap_std
                    x = torch.tensor(feat, dtype=torch.float32).reshape(
                        1, args.seq_len, 1)
                    raw = model.network(x)
                    if output_norm == 'window':
                        # target: per-window EncoderNormalizer, inverted by
                        # the model's own transform_output.
                        scale = max(float(window.std()), 1e-6)
                        target_scale = torch.tensor(
                            [[float(window.mean()), scale]],
                            dtype=torch.float32)
                        pred_t = float(
                            model.transform_output(raw, target_scale).reshape(-1)[0])
                    elif output_norm == 'global_minmax':
                        # Deliberately different inverse: global target
                        # min/max.  This is a diagnostic, not a trained path.
                        pred_t = float(raw.reshape(-1)[0]) * (
                            target_max - target_min) + target_min
                    else:
                        raise ValueError(output_norm)
                preds_t.append(pred_t)
                preds_ah.append(pred_t * args.rated)
                window = np.append(window[1:], pred_t)
        return np.asarray(preds_ah), np.asarray(preds_t)

    def summarize(preds_ah):
        below = np.where(preds_ah <= threshold)[0]
        first_below = args.sp + int(below[0]) + 1 if len(below) else None
        max_idx = int(np.argmax(preds_ah))
        return {
            'initial_prediction_ah': float(preds_ah[0]),
            'max_prediction_ah': float(preds_ah[max_idx]),
            'max_prediction_cycle': args.sp + max_idx + 1,
            'final_prediction_ah': float(preds_ah[-1]),
            'min_prediction_ah': float(preds_ah.min()),
            'threshold_crossed': first_below is not None,
            'first_below_cycle': first_below,
        }

    window_ah, window_t = simulate('model', 'window')
    global_ah, global_t = simulate('model', 'global_minmax')
    persist_ah, persist_t = simulate('persistence', 'window')
    base = summarize(window_ah)
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
        'target_min': target_min,
        'target_max': target_max,
        'input_scaler_mean': cap_mean,
        'input_scaler_std': cap_std,
        'input_normalization': 'min-max (training batteries) then StandardScaler, as in TimeSeriesDataSet',
        'target_normalization': 'per-window EncoderNormalizer (mean/std)',
        'start_cycle': args.sp + 1,
        'end_cycle': args.sp + args.steps,
        'initial_prediction_ah': base['initial_prediction_ah'],
        'max_prediction_ah': base['max_prediction_ah'],
        'max_prediction_cycle': base['max_prediction_cycle'],
        'final_prediction_ah': base['final_prediction_ah'],
        'min_prediction_ah': base['min_prediction_ah'],
        'threshold_crossed': base['threshold_crossed'],
        'first_below_cycle': base['first_below_cycle'],
        'controls': {
            'persistence': summarize(persist_ah),
            'global_minmax_inverse': summarize(global_ah),
        },
        'trajectories_ah': {
            'model_window': window_ah.tolist(),
            'model_global_minmax': global_ah.tolist(),
            'persistence': persist_ah.tolist(),
        },
        'note': 'Closed-loop recursion with capacity-only input. The main trajectory uses the training normalization; the other two are controls: a persistence forecast and a deliberately different global min-max inverse. They test whether the flat trajectory is specific to the model or is forced by the protocol.',
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
    cap = data[args.battery]['Capacity'].values  # already in Ah
    cycles = np.arange(1, len(cap) + 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(cycles, cap, color='#1f77b4', lw=1.1, label='Measurement')
    pred_cycles = np.arange(args.sp + 1, args.sp + 1 + args.steps)
    ax.plot(pred_cycles, window_ah, color='#d62728', lw=1.4,
            label='Model, window inverse')
    ax.plot(pred_cycles, persist_ah, color='#2ca02c', lw=1.1, ls='--',
            label='Persistence (last value)')
    # The global min-max inverse is reported in the JSON/text only: it can
    # produce negative or oscillating values because it is not the inverse
    # used during training, so plotting it would compress the informative
    # 0.95-1.0 Ah region.
    ax.axhline(threshold, ls='--', lw=0.9, color='k',
               label='EOL threshold ({:.2f} Ah)'.format(threshold))
    ax.axvline(args.sp, ls=':', lw=0.8, color='gray')
    ax.annotate('window-inverse final {:.3f} Ah'.format(
                    result['final_prediction_ah']),
                xy=(pred_cycles[-1], result['final_prediction_ah']),
                xytext=(args.sp + 180, result['final_prediction_ah'] + 0.06),
                fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Capacity (Ah)')
    ax.set_title('Closed-loop controls ({}, SP={})'.format(
        args.battery, args.sp))
    ax.set_xlim(0, args.sp + args.steps + 20)
    ax.set_ylim(0.25, 1.15)
    ax.grid(alpha=0.3)
    ax.legend(frameon=True, framealpha=0.92, edgecolor='none',
              loc='center left', bbox_to_anchor=(0.01, 0.40), fontsize=7.5)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_figure)), exist_ok=True)
    fig.savefig(args.output_figure)
    plt.close(fig)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('saved', args.output_json)
    print('saved', args.output_figure)


if __name__ == '__main__':
    main()