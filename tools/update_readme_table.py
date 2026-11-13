# -*- coding: utf-8 -*-
"""把 results/summary_metrics_agg.csv 中 PatchFormer 主结果写进 README 表格。"""
import io
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
agg = pd.read_csv(os.path.join(ROOT, 'results', 'summary_metrics_agg.csv'))

rows = []
for test, label in (('CS2_35', 'CALCE I (CS2_35)'), ('CS2_37', 'CALCE II (CS2_37)')):
    for sp in (300, 400, 500):
        r = agg[(agg.model == 'PatchFormer') & (agg.test_name == test) & (agg.sp == sp)]
        if r.empty:
            rows.append('| {} | PatchFormer | {} |  |  |  |  |'.format(label, sp))
            continue
        r = r.iloc[0]
        rows.append('| {} | PatchFormer | {} | {:.4f} | {:.4f} | {:.4f} | {:.0f} |'.format(
            label, sp, r.MAE, r.RMSE, r.r2, r.AE))
table = ('| 数据集 | 模型 | SP | MAE | RMSE | R² | AE |\n'
         '|---|---|---|---|---|---|---|\n' + '\n'.join(rows))

p = os.path.join(ROOT, 'README.md')
s = io.open(p, encoding='utf-8').read()
start = s.index('| 数据集 | 模型 | SP | MAE | RMSE | R² | AE |')
end = s.index('完整对比见论文表 3~表 8')
s = s[:start] + table + '\n\n' + s[end:]
io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('README 结果表已更新')