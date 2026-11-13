# -*- coding: utf-8 -*-
"""汇总所有实验结果：results/metrics_*.csv -> results/summary_metrics.csv（论文表格唯一数据源）。

用法（项目根目录）：
    python tools/collect_metrics.py          # 合并 + 打印透视表
"""
import os
import glob
import csv
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

METRIC_KEYS = ['MAE', 'RMSE', 'r2', 'RUL_real', 'RUL_pred', 'AE', 'RE', 'epochs', 'train_seconds']


def read_all():
    rows = []
    for path in sorted(glob.glob(os.path.join(RES, 'metrics_*.csv'))):
        with open(path, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if not r or not r.get('model'):
                    continue
                rows.append(r)
    return rows


def main():
    rows = read_all()
    if not rows:
        print('没有找到 results/metrics_*.csv，先跑实验。')
        return
    # 逐条实验 + 按 (model, test_name, sp) 聚合均值
    out_all = os.path.join(RES, 'summary_metrics.csv')
    with open(out_all, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('逐次实验明细 ->', out_all, '（{} 行）'.format(len(rows)))

    groups = defaultdict(list)
    for r in rows:
        groups[(r['model'], r['test_name'], int(r['sp']))].append(r)

    agg_path = os.path.join(RES, 'summary_metrics_agg.csv')
    with open(agg_path, 'w', newline='', encoding='utf-8-sig') as f:
        fields = ['model', 'test_name', 'sp', 'n_runs'] + METRIC_KEYS
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (model, test_name, sp), sel in sorted(groups.items()):
            row = {'model': model, 'test_name': test_name, 'sp': sp, 'n_runs': len(sel)}
            for k in METRIC_KEYS:
                vals = [float(s[k]) for s in sel if s.get(k) not in (None, '')]
                row[k] = round(sum(vals) / len(vals), 6) if vals else ''
            w.writerow(row)
    print('按 (模型, 电池, SP) 聚合 ->', agg_path)

    # 打印一个便于直接抄进论文的表
    print()
    header = '{:<26s}{:>8s}{:>6s}{:>10s}{:>10s}{:>10s}{:>8s}{:>8s}{:>8s}'.format(
        'model', 'battery', 'SP', 'MAE', 'RMSE', 'R2', 'RUL_t', 'RUL_p', 'AE')
    print(header)
    print('-' * len(header))
    for (model, test_name, sp), sel in sorted(groups.items()):
        def m(k):
            vals = [float(s[k]) for s in sel if s.get(k) not in (None, '')]
            return sum(vals) / len(vals) if vals else float('nan')
        print('{:<26s}{:>8s}{:>6d}{:>10.4f}{:>10.4f}{:>10.4f}{:>8.0f}{:>8.0f}{:>8.0f}'.format(
            model, test_name, sp, m('MAE'), m('RMSE'), m('r2'), m('RUL_real'), m('RUL_pred'), m('AE')))


if __name__ == '__main__':
    main()