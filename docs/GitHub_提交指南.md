# GitHub 提交指南（Windows）

作业要求：论文中附上代码与权重（baseline 与提出方法），并给出 GitHub 链接。按下面步骤操作即可。

## 1. 安装 Git

1. 打开 <https://git-scm.com/download/win>，下载 **64-bit Git for Windows Setup**。
2. 一路“下一步”安装（默认选项即可）。
3. 安装完成后**重新打开** PowerShell / 终端，执行：

```powershell
git --version
```

能看到版本号（如 `git version 2.47.0`）即安装成功。

## 2. 配置身份（只需一次）

```powershell
git config --global user.name "你的姓名"
git config --global user.email "你的邮箱"
```

## 3. 初始化仓库并提交

在代码根目录 `C:\Users\86136\Desktop\computer\02_代码` 下打开 PowerShell：

```powershell
cd "C:\Users\86136\Desktop\computer\02_代码"
git init
git add .
git status          # 检查：不应出现 data/ 下的原始 xlsx、results_*/、logs/、lightning_logs/
git commit -m "PatchFormer RUL reproduction: code, weights and results"
```

> `.gitignore` 已排除原始数据（xlsx 约 227 MB）、训练中间目录 `results_*/`、日志与临时文件；
> 保留 `data/CALCE data/CALCE_Data.npy`（0.19 MB）、`data/TJU data/Dataset_3_NCM_NCA_battery_1C.npy`
> （0.4 MB）、`results/` 下的预测结果与指标 CSV、`weights/` 下的最优权重。论文在 `01_论文/`，不随代码仓库提交。

## 4. GitHub 仓库

仓库已创建：<https://github.com/ypc123321/PatchFormer>（Public）。若需要重建，登录 <https://github.com>，右上角 **+ → New repository**，名称可用 `PatchFormer`，可见性选 **Public**，不要勾选 “Add a README file”，然后 **Create repository**。

## 5. 推送

```powershell
git branch -M main
git remote add origin https://github.com/ypc123321/PatchFormer.git
git push -u origin main
```

首次推送会弹出 GitHub 登录窗口（浏览器授权）；若提示密码认证失败，请使用
**Personal Access Token** 代替密码：GitHub 头像 → Settings → Developer settings →
Personal access tokens → Tokens (classic) → Generate new token（勾选 `repo`），
生成后把 token 当作密码粘贴。

## 6. 提交后自检

- 浏览器打开仓库，确认能看到：`README.md`、`RUL_Prediction_benchmark.py`、
  `tools/`、`figures/`、`weights/`、`results/summary_metrics_agg.csv`、
  `results/verification_report.csv`，以及两个预处理数据 npy。
- 确认 `weights/` 下有 15 个 `best.ckpt`：PatchFormer 3 个（CS2_35/CS2_37/CY25_1）、
  三个消融各 1 个、PatchTST/iTransformer/ModernTCN 各 3 个。
- 把仓库链接替换进论文首页与附录 A 的占位处，重新导出 PDF（论文是静态 DOCX，不存在自动生成脚本）。

## 7. 常见问题

| 问题 | 处理 |
|---|---|
| `git: command not found` | Git 未安装成功或终端未重启，重装后再试 |
| 推送速度极慢 / 失败 | 检查是否误把 `data/` 原始数据加入暂存区（`git status` 查看），必要时 `git rm -r --cached data` 后重新提交 |
| 推送体积过大（>100 MB） | 说明有未忽略的大文件，执行 `git count-objects -vH` 检查后清理并重新提交 |
| 老师需要打不开 `.ckpt` | 权重为 PyTorch Lightning checkpoint，可用 `torch.load(path, map_location='cpu')` 读取；如需通用格式可另存 `state_dict` 为 `weights/<model>.pth`（见 `tools/export_weights.py` 说明） |