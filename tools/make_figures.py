# -*- coding: utf-8 -*-
"""生成课程论文所需全部图表（表格数据另由 tools/collect_metrics.py 汇总为 CSV）。

用法（项目根目录）：
    python tools/make_figures.py            # 生成所有当前数据支持的图
    python tools/make_figures.py fig1 fig2  # 只生成指定图

已实现：
    fig1  四节 CALCE 电池容量退化曲线（含 0.77Ah 失效线与预测起点）
    fig2  PatchFormer 总体结构示意图
    fig3  各预测起点（SP300/400/500）的容量预测曲线
    fig4  消融实验 MAE/RMSE 柱状图
    fig5  预测起点 SP 对 MAE 与 RUL 绝对误差的影响
    fig6  容量再生现象放大 + 预测残差分布
数据缺失的图会自动跳过并给出提示，不会报错。
"""
import os
import sys
import glob
import csv
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']


def save(fig, name):
    path = os.path.join(FIG_DIR, name + '.png')
    fig.savefig(path)
    plt.close(fig)
    print('saved', path)


def load_calce():
    path = os.path.join(ROOT, 'data', 'CALCE data', 'CALCE_Data.npy')
    return np.load(path, allow_pickle=True).item()


# ------------------------------------------------------------------ fig1
def fig1_capacity(batteries, rated=1.1, threshold=0.77):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, (name, df) in enumerate(batteries.items()):
        cap = df['Capacity'].values
        ax.plot(np.arange(1, len(cap) + 1), cap, lw=1.2, color=COLORS[i % len(COLORS)],
                label='{} ({} cycles)'.format(name, len(cap)))
    ax.axhline(threshold, ls='--', lw=1.0, color='k')
    ax.text(10, threshold + 0.02, 'EOL threshold ({:.2f} Ah)'.format(threshold), fontsize=8)
    for sp in (300, 400, 500):
        ax.axvline(sp, ls=':', lw=0.8, color='gray')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Capacity (Ah)')
    ax.set_xlim(0, 1000)
    ax.legend(loc='upper right', frameon=False)
    ax.grid(alpha=0.3)
    save(fig, 'fig1_capacity_curves')


# ------------------------------------------------------------------ fig2
def fig2_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    def box(x, y, w, h, text, fc, ec='#333333'):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=0.12',
                           fc=fc, ec=ec, lw=1.0)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=10,
                                     lw=1.0, color='#333333'))

    box(0.2, 2.0, 1.5, 1.0, 'Capacity\nsequence\n(64 cycles)', '#e8f1fa')
    box(2.1, 2.0, 1.5, 1.0, 'Patch\nembedding\n(P=2, S=2)', '#e8f1fa')
    box(4.0, 3.0, 2.0, 0.8, 'Intra-patch attention', '#dff2e6')
    box(4.0, 1.2, 2.0, 0.8, 'Inter-patch attention', '#dff2e6')
    box(6.4, 2.0, 1.5, 1.0, 'Feature\nattention\n(SE)', '#fdf1dd')
    box(8.2, 2.0, 1.5, 1.0, 'Flatten +\nFC\n$\\rightarrow$ RUL', '#f5e6f4')

    arrow(1.7, 2.5, 2.1, 2.5)
    arrow(3.6, 2.5, 4.0, 3.4)
    arrow(3.6, 2.5, 4.0, 1.6)
    arrow(6.0, 3.4, 6.4, 2.6)
    arrow(6.0, 1.6, 6.4, 2.4)
    arrow(7.9, 2.5, 8.2, 2.5)
    ax.text(5.0, 4.35, 'Dual patch attention network (DPAN)', ha='center', fontsize=9)
    ax.text(5.0, 0.6, 'PatchFormer', ha='center', fontsize=11, weight='bold')
    save(fig, 'fig2_architecture')


# ------------------------------------------------------------------ 数据读取工具
def read_metrics():
    """合并 results/metrics_*.csv 为列表（列名小写化）。"""
    rows = []
    for path in glob.glob(os.path.join(ROOT, 'results', 'metrics_*.csv')):
        with open(path, newline='', encoding='utf-8') as f:
            rows.extend(list(csv.DictReader(f)))
    for r in rows:
        for k in ('MAE', 'RMSE', 'r2', 'RE'):
            r[k] = float(r[k])
        for k in ('sp', 'run', 'epochs', 'RUL_real', 'RUL_pred', 'AE'):
            r[k] = int(float(r[k]))
    return rows


def load_pred(test_name, model):
    import torch
    path = os.path.join(ROOT, 'results', 'RUL_{}_{}.pth'.format(test_name, model))
    if not os.path.exists(path):
        return None, path
    return torch.load(path, map_location='cpu'), path


def true_curve(test_name, rated=1.1):
    path = os.path.join(ROOT, 'results', 'Capacity_{}_real_data.pth'.format(test_name))
    import torch
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu')


# ------------------------------------------------------------------ fig3
def fig3_predictions(metrics, test_name='CS2_35', model='PatchFormer'):
    data, path = load_pred(test_name, model)
    if data is None:
        print('skip fig3: missing', path)
        return
    true_cap = true_curve(test_name)
    sps = sorted(int(k[2:]) for k in data.keys())
    fig, axes = plt.subplots(1, len(sps), figsize=(12, 3.4), sharey=True)
    if len(sps) == 1:
        axes = [axes]
    for ax, sp in zip(axes, sps):
        pred = np.asarray(data['SP{}'.format(sp)][0])
        x0 = sp
        ax.plot(np.arange(x0, x0 + len(pred)), pred, lw=1.2, color='#d62728', label='Prediction')
        if true_cap is not None:
            # Predictions start at the prediction point itself (cycle SP).
            seg = true_cap[x0 - 1: x0 - 1 + len(pred)]
            ax.plot(np.arange(x0, x0 + len(seg)), seg, lw=1.0, color='#1f77b4', label='Measurement')
        ax.axhline(0.77, ls='--', lw=0.8, color='k')
        ax.set_title('SP = {}'.format(sp))
        ax.set_xlabel('Cycle')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Capacity (Ah)')
    axes[0].legend(frameon=False, loc='upper right')
    fig.suptitle('RUL prediction of {} (test battery {}), model {}'.format(test_name, test_name, model), y=1.02)
    save(fig, 'fig3_predictions_{}'.format(test_name))


# ------------------------------------------------------------------ fig4
ABLATION_ORDER = ['PatchFormer', 'PatchFormer_no_DPAN', 'PatchFormer_no_inter_PA', 'PatchFormer_no_intra_PA']
ABLATION_LABEL = {'PatchFormer': 'PatchFormer', 'PatchFormer_no_DPAN': 'w/o DPAN',
                  'PatchFormer_no_inter_PA': 'w/o inter-PA', 'PatchFormer_no_intra_PA': 'w/o intra-PA'}


def fig4_ablation(metrics, test_name='CS2_35'):
    rows = [r for r in metrics if r['model'] in ABLATION_ORDER and r['test_name'] == test_name]
    if not rows:
        print('skip fig4: no ablation metrics')
        return
    models = [m for m in ABLATION_ORDER if any(r['model'] == m for r in rows)]
    sps = sorted({r['sp'] for r in rows})
    x = np.arange(len(models))
    width = 0.8 / max(len(sps), 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), sharex=True)
    for j, metric in enumerate(('MAE', 'RMSE')):
        ax = axes[j]
        for i, sp in enumerate(sps):
            vals = []
            for model in models:
                sel = [r[metric] for r in rows
                       if r['model'] == model and r['sp'] == sp]
                vals.append(np.mean(sel) if sel else np.nan)
            bars = ax.bar(x - 0.4 + width * (i + 0.5), vals, width,
                          label='SP{}'.format(sp), color=COLORS[i % len(COLORS)])
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v, '{:.4f}'.format(v),
                            ha='center', va='bottom', fontsize=6.5, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([ABLATION_LABEL[m] for m in models], fontsize=8)
        ax.set_ylabel('{} (Ah)'.format(metric))
        ax.set_title(metric)
        ax.grid(axis='y', alpha=0.3)
    axes[1].legend(frameon=False, title='Starting point', fontsize=7.5)
    fig.suptitle('Ablation study on {} (grouped by starting point)'.format(test_name), y=1.02)
    save(fig, 'fig4_ablation')


# ------------------------------------------------------------------ fig5
def fig5_sp_effect(metrics, test_name='CS2_35'):
    rows = [r for r in metrics if r['test_name'] == test_name]
    if not rows:
        print('skip fig5: no metrics')
        return
    models = sorted({r['model'] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for i, model in enumerate(models):
        sps = sorted({r['sp'] for r in rows if r['model'] == model})
        mae, ae = [], []
        for sp in sps:
            sel = [r for r in rows if r['model'] == model and r['sp'] == sp]
            mae.append(np.mean([r['MAE'] for r in sel]))
            ae.append(np.mean([r['AE'] for r in sel]))
        axes[0].plot(sps, mae, marker='o', lw=1.2, color=COLORS[i % len(COLORS)], label=model)
        axes[1].plot(sps, ae, marker='s', lw=1.2, color=COLORS[i % len(COLORS)], label=model)
    for ax, title, ylabel in ((axes[0], 'MAE', 'MAE (Ah)'), (axes[1], 'RUL absolute error', 'Cycles')):
        ax.set_xlabel('Starting prediction point (cycle)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].legend(frameon=False, fontsize=7.5)
    save(fig, 'fig5_sp_effect')


# ------------------------------------------------------------------ fig6
def fig6_regeneration(batteries, test_name='CS2_35', rated=1.1, sp=300):
    if test_name not in batteries:
        print('skip fig6: battery not found')
        return
    pred_data, pred_path = load_pred(test_name, 'PatchFormer')
    no_data, no_path = load_pred(test_name, 'PatchFormer_no_inter_PA')
    if pred_data is None or no_data is None:
        print('skip fig6: missing prediction files', pred_path, no_path)
        return
    df = batteries[test_name]
    cap = df['Capacity'].values
    cycles = np.arange(1, len(cap) + 1)
    win = 25
    trend = np.convolve(cap, np.ones(win) / win, mode='same')
    valid = slice(win, len(cap) - win)
    resid = np.zeros_like(cap)
    resid[valid] = cap[valid] - trend[valid]

    pred = np.asarray(pred_data['SP{}'.format(sp)][0])
    pred_no = np.asarray(no_data['SP{}'.format(sp)][0])
    pred_cycles = np.arange(sp, sp + len(pred))
    # Regeneration points: rebound > 0.01 Ah within 10 cycles.
    reg = np.where((cap[10:] - cap[:-10]) > 0.01)[0] + 10
    reg = reg[(reg >= sp) & (reg <= sp - 1 + len(pred))]
    idx = reg - sp + 1
    mae_full = float(np.mean(np.abs(pred[idx] - cap[reg])))
    mae_no = float(np.mean(np.abs(pred_no[idx] - cap[reg])))

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.2),
                             gridspec_kw={'height_ratios': [1.35, 1]})
    ax = axes[0]
    ax.plot(cycles, cap, lw=0.9, color='#1f77b4', label='Measurement')
    ax.plot(cycles, trend, lw=1.2, color='#ff7f0e', label='25-cycle moving average')
    ax.axhline(0.77, ls='--', lw=0.8, color='k', label='EOL threshold (0.77 Ah)')
    ax.axvline(sp, ls=':', lw=0.8, color='gray')
    ax.set_ylabel('Capacity (Ah)')
    ax.set_title('Capacity regeneration of {}'.format(test_name))
    ax.set_xlim(0, len(cap))
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')
    ax.grid(alpha=0.3)

    ax = axes[1]
    lo, hi = 330, 375
    m = (cycles >= lo) & (cycles <= hi)
    ax.plot(cycles[m], cap[m], 'o-', ms=2.0, lw=0.8, color='#1f77b4', label='Measurement')
    pm = (pred_cycles >= lo) & (pred_cycles <= hi)
    ax.plot(pred_cycles[pm], pred[pm], lw=1.3, color='#d62728',
            label='PatchFormer (reg. MAE {:.4f})'.format(mae_full))
    ax.plot(pred_cycles[pm], pred_no[pm], lw=1.1, color='#2ca02c', ls='--',
            label='w/o inter-PA (reg. MAE {:.4f})'.format(mae_no))
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Capacity (Ah)')
    ax.set_title('Zoom: cycles {}-{}'.format(lo, hi))
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    ax.grid(alpha=0.3)
    save(fig, 'fig6_regeneration_{}'.format(test_name))


def main():
    wanted = sys.argv[1:] or ['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6']
    batteries = load_calce()
    metrics = read_metrics()

    if 'fig1' in wanted:
        fig1_capacity(batteries)
    if 'fig2' in wanted:
        fig2_architecture()
    if 'fig3' in wanted:
        fig3_predictions(metrics)
    if 'fig4' in wanted:
        fig4_ablation(metrics)
    if 'fig5' in wanted:
        fig5_sp_effect(metrics)
    if 'fig6' in wanted:
        fig6_regeneration(batteries)


if __name__ == '__main__':
    main()