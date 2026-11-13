# -*- coding: utf-8 -*-
"""把每次实验的最优权重（best_model_path）复制到 weights/<model>/<test>/ 下，便于随仓库发布。

用法（项目根目录）：
    python tools/export_weights.py
"""
import os
import csv
import glob
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'weights')


def main():
    seen = set()
    copied = 0
    for path in sorted(glob.glob(os.path.join(RES, 'metrics_*.csv'))):
        with open(path, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                bmp = (r.get('best_model_path') or '').strip()
                if not bmp:
                    continue
                src = bmp if os.path.isabs(bmp) else os.path.join(ROOT, bmp)
                if not os.path.exists(src):
                    print('missing ckpt:', src)
                    continue
                model, test = r['model'], r['test_name']
                dst_dir = os.path.join(OUT, model, test)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, 'best.ckpt')
                if dst in seen:
                    continue
                seen.add(dst)
                shutil.copy2(src, dst)
                print('copied {} -> {}'.format(src, dst))
                copied += 1
    print('done, {} weight file(s) -> {}'.format(copied, OUT))


if __name__ == '__main__':
    main()