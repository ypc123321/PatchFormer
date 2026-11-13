# 权重说明

本目录保存 15 个最优 checkpoint，每个“模型 × 测试电池”一个文件。文件路径为：

```
weights/<model>/<test_name>/best.ckpt
```

## 为什么一个权重可以覆盖三个预测起点

训练集排除测试电池，预测起点 SP 只改变测试集的截取位置，不改变训练数据与模型参数。因此同一个 checkpoint 可以分别在 SP=300/400/500（TJU 为 200/300/400）上评估。`tools/verify_reproduction.py` 已逐一复算 45 个组合，结果与 `results/summary_metrics_agg.csv` 一致（最大绝对差 1e-6）。

## 清单

| 模型 | 电池 | 数据集 | 训练轮数 | SP300/200 MAE (Ah) |
|---|---|---|---:|---:|
| PatchFormer | CS2_35 | CALCE | 88 | 0.005825 |
| PatchFormer | CS2_37 | CALCE | 61 | 0.005469 |
| PatchFormer | CY25_1 | TJU | 13 | 0.001504 |
| PatchFormer_no_DPAN | CS2_35 | CALCE | 67 | 0.005768 |
| PatchFormer_no_inter_PA | CS2_35 | CALCE | 80 | 0.007315 |
| PatchFormer_no_intra_PA | CS2_35 | CALCE | 78 | 0.005715 |
| PatchTST | CS2_35 | CALCE | 60 | 0.018547 |
| PatchTST | CS2_37 | CALCE | 60 | 0.012000 |
| PatchTST | CY25_1 | TJU | 39 | 0.010046 |
| iTransformer | CS2_35 | CALCE | 60 | 0.014789 |
| iTransformer | CS2_37 | CALCE | 60 | 0.008297 |
| iTransformer | CY25_1 | TJU | 23 | 0.001734 |
| ModernTCN | CS2_35 | CALCE | 60 | 0.011964 |
| ModernTCN | CS2_37 | CALCE | 57 | 0.007649 |
| ModernTCN | CY25_1 | TJU | 47 | 0.001502 |

训练轮数为实际提前停止时的 epoch（`metrics_*.csv` 的 `epochs` 列），不是最大预算。