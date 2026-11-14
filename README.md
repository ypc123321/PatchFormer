# PatchFormer 锂电池剩余寿命（RUL）预测复现

> GitHub 仓库：<https://github.com/ypc123321/PatchFormer>

本仓库是基于 **PatchFormer**（Liu et al., *Journal of Power Sources* 631 (2025) 236187）官方开源代码的**课程大作业复现项目**：在 CALCE 与 TJU 公开数据上完成数据预处理、训练、容量预测与 RUL 换算的端到端复现，并补充了消融实验、主流基线对比与图表生成脚本。

> 原始官方仓库见 `README_official.md`。本仓库保留其网络结构与数据处理逻辑，新增统一实验入口、批量实验脚本、指标汇总、绘图、权重复核与闭环递归探针。

## 目录结构

```
RUL_Prediction_benchmark.py   统一实验入口（本提交实际使用 7 个模型）
RUL_Prediction_PatchFormer*.py 原始单模型脚本（主体与三个消融变体）
CALCEDataPreProcess.py        CALCE 数据预处理（库仑积分、异常剔除、归一化）
TJUDataPreProcess.py          TJU 数据预处理
ModelsModify/                 模型实现（PatchFormer 及各类基线封装）
tools/
  run_experiments.py          顺序执行 15 个实验（断点续跑）
  collect_metrics.py          汇总 results/metrics_*.csv
  make_figures.py             生成论文全部图表（figures/）
  export_weights.py           把最优权重复制到 weights/
  verify_reproduction.py      用 weights/ 复算 15×3 个指标并与 CSV 对账
  recursive_probe.py          闭环递归外推压力测试（论文4.10 节 / 图 7）
  smoke_models.py             模型的一步冒烟测试
data/CALCE data/              CALCE 预处理后的容量序列（CALCE_Data.npy）
data/TJU data/                TJU 预处理后的容量序列（Dataset_3_NCM_NCA_battery_1C.npy）
results/                      预测结果（RUL_*.pth）、指标 CSV 与核验报告
weights/                      训练好的最优权重（.ckpt，说明见 weights/README.md）
figures/                      论文图表
```

本提交不包含论文自动生成脚本；论文 DOCX/PDF 为最终交付物，图表由 `tools/make_figures.py` 生成。

## 环境

- 实测环境为 conda 环境 `pf_rul`：
  - Python 3.9.25，PyTorch 2.0.1 + CUDA 11.8
  - PyTorch Lightning 1.9.5，PyTorch Forecasting 0.10.3
  - numpy 1.21.6，pandas 1.5.3，scikit-learn 1.1.3
- 单卡 RTX 3060 Laptop（6 GB）：主体模型单轮训练约 8 s（21 个 batch），显存峰值约 161 MiB。
- 依赖见 `requirements.txt`；官方 conda 环境导出见 `patchformer.yaml`。
- 代码会自动检测 CUDA；无 GPU 时回退到 CPU（速度明显变慢，但可运行）。

## 数据

- CALCE：`data/CALCE data/CALCE_Data.npy`，四节电池 CS2_35~CS2_38，按 70% 额定容量（0.77 Ah）标定寿命终点。
- 重新生成 CALCE 预处理数据：`python CALCEDataPreProcess.py --test_name CS2_35`
- TJU：`data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy`（三节 NCM+NCA 电池 CY25_1~CY25_3），失效阈值 1.75 Ah（2.5 Ah 的 70%）；实验以 CY25_2/3 训练、CY25_1 测试，SP 取 200/300/400。
- NASA：参见 `NASADataPreProcess.py`，数据需自行下载（本仓库未包含）。

## 复现实验

```bash
# 1) 单次实验（主体模型，CALCE I，三个预测起点，100 epochs）
python RUL_Prediction_benchmark.py --model PatchFormer --dataset CALCE \
    --data_dir "data/CALCE data/" --test_name CS2_35 --Rated_Capacity 1.1 \
    --count 1 --max_epochs 100 --start_point_list 300,400,500

# 1b) TJU 数据集补充验证（CY25_1 测试，SP 200/300/400）
python RUL_Prediction_benchmark.py --model PatchFormer --dataset TJU \
    --data_dir "data/TJU data/" --test_name CY25_1 --Rated_Capacity 2.5 \
    --root_dir TJU_RUL_prediction_sl_64 --count 1 --max_epochs 100 \
    --start_point_list 200,300,400

# 2) 全部 15 个实验（主结果 + 消融 + 基线），已完成的会自动跳过
python tools/run_experiments.py

# 3) 汇总指标 / 画图 / 导出权重
python tools/collect_metrics.py
python tools/make_figures.py
python tools/export_weights.py

# 4) 复现核验（15 个权重 × 3 个 SP，结果写入 results/verification_report.csv）
python tools/verify_reproduction.py

# 5) 闭环递归探针（论文4.10 节，输出 results/recursive_probe.json 与 figures/fig7_recursive.png）
python tools/recursive_probe.py --battery CS2_35 --sp 300
```

可选模型（`--model`）：`PatchFormer`、`PatchFormer_no_DPAN`、`PatchFormer_no_inter_PA`、`PatchFormer_no_intra_PA`、`PatchTST`、`iTransformer`、`ModernTCN`，以及代码中保留但未纳入本提交实验的 `TimesNet`、`TimeMixer`、`MambaSimple`、`Autoformer`、`FEDformer`、`PathFormer`。

## 主要结果

复现模型的实验结果保存在 `results/`，下表由 `tools/collect_metrics.py` 生成（以实际运行结果为准）：

| 数据集 | 模型 | SP | MAE | RMSE | R² | AE |
|---|---|---|---|---|---|---|
| CALCE I (CS2_35) | PatchFormer | 300 | 0.0058 | 0.0127 | 0.9963 | 1 |
| CALCE I (CS2_35) | PatchFormer | 400 | 0.0066 | 0.0139 | 0.9955 | 0 |
| CALCE I (CS2_35) | PatchFormer | 500 | 0.0077 | 0.0155 | 0.9937 | 0 |
| CALCE II (CS2_37) | PatchFormer | 300 | 0.0055 | 0.0089 | 0.9981 | 3 |
| CALCE II (CS2_37) | PatchFormer | 400 | 0.0057 | 0.0094 | 0.9979 | 3 |
| CALCE II (CS2_37) | PatchFormer | 500 | 0.0061 | 0.0101 | 0.9975 | 3 |
| TJU (CY25_1) | PatchFormer | 200 | 0.0015 | 0.0023 | 0.9998 | 2 |
| TJU (CY25_1) | PatchFormer | 300 | 0.0017 | 0.0025 | 0.9996 | 3 |
| TJU (CY25_1) | PatchFormer | 400 | 0.0025 | 0.0033 | 0.9990 | 4 |

完整对比见论文表 4~表 9 与 `results/summary_metrics_agg.csv`（含 PatchTST / iTransformer / ModernTCN 及三个消融变体）。

## 关键口径说明

- **RUL 换算不是闭环迭代。** 测试阶段为 SP 之后的每个真实历史窗口构造输入，模型只做单步容量预测；这些单步预测按循环顺序串成序列，首次低于 0.77 Ah（TJU 为 1.75 Ah）的位置换算为 PRUL。代码见 `RUL_Prediction_benchmark.py` 的 `best_model.predict(test_dataloader)` 与 `rul_value_error(...)`。
- **表 8 的低 AE 只反映单步容量精度**，不能解释为长期外推能力。论文4.10 节与 `tools/recursive_probe.py` 给出闭环递归压力测试：把预测值回填窗口后，CS2_35/SP300 的容量在窗口反归一化下约 0.96 Ah、1200 步内不穿越 0.77 Ah；但 persistence 对照同样不穿越，而全局反归一化轨迹失真，因此当前协议不能单独判定模型本身无外推能力。
- **一个权重覆盖三个 SP。** 训练集排除测试电池，SP 只改变测试集的截取位置；因此 `weights/<model>/<battery>/best.ckpt` 可用于该电池的 SP300/400/500（TJU 为 200/300/400）。`tools/verify_reproduction.py` 已对 45 个“模型×电池×SP”组合逐一复算，全部与 CSV 一致（最大绝对差 1e-6）。
- **训练预算并不完全对齐。** 主模型 100 轮、消融 80 轮、基线 60 轮；这是复现时的实际设置，论文表 2 与第 5 节已如实披露，因此跨模型比较应理解为“各自预算下的结果”，不是严格等算力对比。

## 复现说明

- 数据切分与论文一致：CALCE I 训练 CS2_36/37/38、测试 CS2_35（SP 300/400/500）；CALCE II 训练 CS2_35/36/38、测试 CS2_37；TJU 训练 CY25_2/3、测试 CY25_1（SP 200/300/400）。
- 输入长度 64、patch 长度 2（步长 2）、嵌入维度 16、FAN 头数 8、编码层数 2、FAN 隐层 64、batch 128、Adam、初始学习率 1e-3、梯度裁剪 0.2、损失 SMAPE、早停 patience 10。
- 与原文的差异：本文每组仅训练一次（原文对每个 SP 重复多次取平均、200 轮，并用 Optuna 搜索超参数），因此绝对数值与原文存在偏差；本文不声称与原论文数值一致，只报告本机真实运行结果。
- 已修复的问题：`RUL_Prediction_PatchFormer.py` 中结果字典硬编码 `SP300/SP400/SP500`（改为按实际 SP 动态生成）；Optuna 与 SQLAlchemy 2.x 的 sqlite 兼容问题（见 `ModelsModify/optimize.py`）；显存溢出问题（收窄搜索空间、trial 内释放显存、`catch=(RuntimeError,)`）；绘图脚本中容量重复乘额定容量与预测段错位一个循环的问题。

## 复现核验

```bash
python tools/verify_reproduction.py
python tools/recursive_probe.py --battery CS2_35 --sp 300
```

`verify_reproduction.py` 会加载 `weights/` 下全部 15 个 checkpoint，重建与训练时一致的 `TimeSeriesDataSet`，对三个 SP 复算 MAE/RMSE/R²/AE，并与 `results/summary_metrics_agg.csv` 对账；`recursive_probe.py` 输出闭环轨迹 JSON 与图 7。

## 引用

```bibtex
@article{liu2025patchformer,
  title   = {PatchFormer: A novel patch-based transformer for accurate remaining useful life prediction of lithium-ion batteries},
  author  = {Liu, Lei and Huang, Jiahui and Zhao, Hongwei and Li, Tianqi and Li, Bin},
  journal = {Journal of Power Sources},
  volume  = {631},
  pages   = {236187},
  year    = {2025}
}
```