# -*- coding: utf-8 -*-
"""顺序执行全部 15 个实验（可断点续跑：结果 pth 已存在则跳过）。

15 个任务 = 3 个主模型（CALCE I / CALCE II / TJU）
          + 3 个消融（CALCE I）
          + 9 个基线（3 模型 × 3 个测试电池）。

训练预算按论文表 2 披露：主模型 100、消融 80、基线 60 轮。
CALCE 的预测起点为 300/400/500，TJU 为 200/300/400；CALCE 与 TJU 的
额定容量分别为 1.1 Ah 与 2.5 Ah。

用法（项目根目录）：
    python tools/run_experiments.py               # 全跑，已完成的自动跳过
    python tools/run_experiments.py --only main   # 只跑主模型
    python tools/run_experiments.py --force       # 重跑（忽略已有 pth）
    python tools/run_experiments.py --dry-run
"""
import os
import sys
import time
import subprocess
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'logs')
os.makedirs(LOGS, exist_ok=True)

CALCE = {
    'dataset': 'CALCE',
    'data_dir': 'data/CALCE data/',
    'rated': 1.1,
    'root_dir': 'CALCE_RUL_prediction_sl_64',
    'start_points': '300,400,500',
}
TJU = {
    'dataset': 'TJU',
    'data_dir': 'data/TJU data/',
    'rated': 2.5,
    'root_dir': 'TJU_RUL_prediction_sl_64',
    'start_points': '200,300,400',
}

JOBS = [
    # group, model, test_name, data spec, max_epochs
    ('main', 'PatchFormer', 'CS2_35', CALCE, 100),
    ('main', 'PatchFormer', 'CS2_37', CALCE, 100),
    ('main', 'PatchFormer', 'CY25_1', TJU, 100),
    ('ablation', 'PatchFormer_no_DPAN', 'CS2_35', CALCE, 80),
    ('ablation', 'PatchFormer_no_inter_PA', 'CS2_35', CALCE, 80),
    ('ablation', 'PatchFormer_no_intra_PA', 'CS2_35', CALCE, 80),
    ('baseline', 'PatchTST', 'CS2_35', CALCE, 60),
    ('baseline', 'iTransformer', 'CS2_35', CALCE, 60),
    ('baseline', 'ModernTCN', 'CS2_35', CALCE, 60),
    ('baseline', 'PatchTST', 'CS2_37', CALCE, 60),
    ('baseline', 'iTransformer', 'CS2_37', CALCE, 60),
    ('baseline', 'ModernTCN', 'CS2_37', CALCE, 60),
    ('baseline', 'PatchTST', 'CY25_1', TJU, 60),
    ('baseline', 'iTransformer', 'CY25_1', TJU, 60),
    ('baseline', 'ModernTCN', 'CY25_1', TJU, 60),
]


def result_path(model, test_name):
    return os.path.join(ROOT, 'results', 'RUL_{}_{}.pth'.format(test_name, model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='all', choices=['all', 'main', 'ablation', 'baseline'])
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    todo = [j for j in JOBS if args.only in ('all', j[0])]
    print('planned jobs: {}'.format(len(todo)))
    t_start = time.time()
    failed = []
    for i, (group, model, test_name, spec, epochs) in enumerate(todo, 1):
        pth = result_path(model, test_name)
        if os.path.exists(pth) and not args.force:
            print('[skip {}/{}] {} on {} (exists: {})'.format(i, len(todo), model, test_name, os.path.basename(pth)))
            continue
        log_path = os.path.join(LOGS, '{}_{}.log'.format(model, test_name))
        cmd = [sys.executable, '-B', 'RUL_Prediction_benchmark.py',
               '--model', model, '--dataset', spec['dataset'],
               '--test_name', test_name, '--data_dir', spec['data_dir'],
               '--Rated_Capacity', str(spec['rated']), '--root_dir', spec['root_dir'],
               '--count', '1', '--max_epochs', str(epochs),
               '--start_point_list', spec['start_points']]
        print('\n=== [{}/{}] {} | {} | {} epochs ==='.format(i, len(todo), model, test_name, epochs))
        print(' '.join(cmd))
        print('log ->', log_path)
        if args.dry_run:
            continue
        t0 = time.time()
        with open(log_path, 'w', encoding='utf-8') as f:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        dt = (time.time() - t0) / 60.0
        ok = proc.returncode == 0 and os.path.exists(pth)
        print('    -> {} in {:.1f} min (returncode={})'.format('OK' if ok else 'FAILED', dt, proc.returncode))
        if not ok:
            failed.append((model, test_name, proc.returncode))
            print('    tail of log:')
            try:
                with open(log_path, encoding='utf-8', errors='replace') as f:
                    print(''.join(f.readlines()[-15:]))
            except Exception as e:
                print('   (cannot read log: {})'.format(e))
    print('\nall done in {:.1f} min, failed: {}'.format((time.time() - t_start) / 60.0, failed))


if __name__ == '__main__':
    main()