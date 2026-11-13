# -*- coding: utf-8 -*-
"""冒烟测试：逐个模型构建 + 单批次前向/反传，确认 13 个模型在 seq_len=64/patch_len=2 下都能跑。
用法（在项目根目录）：python tools/smoke_models.py
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant import get_gpus_memory_info

id, _ = get_gpus_memory_info()
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)

import gc
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer
from pytorch_forecasting.metrics import SMAPE
from CALCEDataPreProcess import BatteryDataProcess

SUBSET = [m for m in sys.argv[1:]] or None

names = ['PatchFormer', 'PatchFormer_no_DPAN', 'PatchFormer_no_inter_PA', 'PatchFormer_no_intra_PA',
         'PatchTST', 'iTransformer', 'ModernTCN', 'TimesNet', 'TimeMixer', 'MambaSimple',
         'Autoformer', 'FEDformer', 'PathFormer']
if SUBSET:
    names = [n for n in names if n in SUBSET]

seq_len, pred_len, patch_len, d_model, bs = 64, 1, 2, 16, 64


class Args:
    test_name = 'CS2_35'
    Rated_Capacity = 1.1
    seq_len = 64
    patch_len = 2
    d_model = 16
    max_epochs = 5
    count = 1
    batch_size = 128
    start_point_list = [300, 400, 500]
    Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
    data_dir = 'data/CALCE data/'


args = Args()
BatteryData = np.load(os.path.join(args.data_dir, 'CALCE_Data.npy'), allow_pickle=True).item()
df_train, df_test, df_all = BatteryDataProcess(BatteryData, args.test_name, 300, args)
mask_len = len(df_train)
training = TimeSeriesDataSet(
    df_train[0:int(0.8 * mask_len)], time_idx="time_idx", target="target", group_ids=['group_id'],
    min_encoder_length=seq_len, max_encoder_length=seq_len,
    min_prediction_length=pred_len, max_prediction_length=pred_len,
    time_varying_known_reals=['Capacity'], time_varying_unknown_reals=['target'],
    target_normalizer=EncoderNormalizer(), add_encoder_length=False)
dl = training.to_dataloader(train=True, batch_size=bs, shuffle=False, num_workers=0, drop_last=True)
x, (y, w) = next(iter(dl))
# 真实训练时 Lightning 会自动把 batch 搬到 GPU，这里手动搬
x = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in x.items()}
y = y.cuda() if torch.is_tensor(y) else y
w = w.cuda() if torch.is_tensor(w) else w

import importlib  # noqa: E402

MODEL_REGISTRY = {
    'PatchFormer': ('ModelsModify.PatchFormer', 'PatchFormerNetModel'),
    'PatchFormer_no_DPAN': ('ModelsModify.PatchFormer_no_DPAN', 'PatchFormer_no_DPANNetModel'),
    'PatchFormer_no_inter_PA': ('ModelsModify.PatchFormer_no_inter_PA', 'PatchFormer_no_inter_PANetModel'),
    'PatchFormer_no_intra_PA': ('ModelsModify.PatchFormer_no_intra_PA', 'PatchFormer_no_intra_PANetModel'),
    'PatchTST': ('ModelsModify.PatchTST', 'PatchTSTNetModel'),
    'iTransformer': ('ModelsModify.iTransformer', 'iTransformerModel'),
    'ModernTCN': ('ModelsModify.ModernTCN', 'ModernTCNModel'),
    'TimesNet': ('ModelsModify.TimesNet', 'TimesNetModel'),
    'TimeMixer': ('ModelsModify.TimeMixer', 'TimeMixerNetModel'),
    'MambaSimple': ('ModelsModify.MambaSimple', 'MambaSimpleNetModel'),
    'Autoformer': ('ModelsModify.Autoformer', 'AutoFormerNetModel'),
    'FEDformer': ('ModelsModify.FEDformer', 'FEDFormerNetModel'),
    'PathFormer': ('ModelsModify.PathFormer', 'PathFormerModel'),
}


def import_model_class(name):
    module_name, class_name = MODEL_REGISTRY[name]
    return getattr(importlib.import_module(module_name), class_name)

ok, bad = [], []
for name in names:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cls = import_model_class(name)
        enc_in = 1
        common = dict(learning_rate=1e-3, loss=SMAPE())
        if name.startswith('PatchFormer'):
            kw = dict(patch_len=patch_len, seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, d_model=d_model,
                      n_heads=8, e_layers=2, dropout=0.1, **common)
        elif name == 'PatchTST':
            kw = dict(patch_len=patch_len, stride=patch_len, seq_len=seq_len, pred_len=pred_len, enc_in=enc_in,
                      d_model=d_model, **common)
        elif name == 'iTransformer':
            kw = dict(input_size=seq_len, output_size=pred_len, d_model=d_model, enc_in=enc_in, n_heads=8,
                      e_layers=2, d_ff=2 * d_model, dropout=0.1, **common)
        elif name == 'ModernTCN':
            kw = dict(enc_in=enc_in, input_size=seq_len, output_size=pred_len, hidden_size=64, P=4, S=2,
                      kernel_sizes=[5, 51], r=2, num_layers=2, **common)
        elif name == 'TimesNet':
            kw = dict(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, dec_in=enc_in, c_out=1, d_model=d_model,
                      e_layers=2, d_ff=2 * d_model, dropout=0.1, **common)
        elif name == 'TimeMixer':
            kw = dict(seq_len=seq_len, label_len=0, pred_len=pred_len, enc_in=enc_in, c_out=1, d_model=d_model,
                      e_layers=2, d_ff=2 * d_model, dropout=0.1, **common)
        elif name == 'MambaSimple':
            kw = dict(seq_len=seq_len, label_len=0, pred_len=pred_len, enc_in=enc_in, dec_in=enc_in, c_out=1,
                      d_model=d_model, e_layers=2, d_ff=2 * d_model, dropout=0.1, **common)
        elif name in ('Autoformer', 'FEDformer'):
            kw = dict(seq_len=seq_len, label_len=0, pred_len=pred_len, enc_in=enc_in, dec_in=enc_in, c_out=1,
                      d_model=d_model, e_layers=2, d_ff=2 * d_model, dropout=0.1, **common)
        elif name == 'PathFormer':
            kw = dict(enc_in=enc_in, seq_len=seq_len, pred_len=pred_len, k=3, patch_size_list=[2, 4, 8, 16], **common)
        t0 = time.time()
        model = cls.from_dataset(training, **kw)
        n_params = model.size()
        trainer = pl.Trainer(fast_dev_run=1, gpus=1, logger=False, gradient_clip_val=0.2,
                             enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(model, train_dataloaders=dl, val_dataloaders=dl)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        print('[OK] {:<24s} params={:>8.1f}k  peak={:>7.1f}MiB  one-batch-fit={:>6.2f}s'.format(
            name, n_params / 1e3, peak, dt))
        ok.append(name)
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print('[FAIL] {:<24s} {}: {}'.format(name, type(e).__name__, str(e)[:300]))
        bad.append((name, type(e).__name__, str(e)[:200]))
        traceback.print_exc(limit=3)

print()
print('OK  :', ok)
print('FAIL:', [b[0] for b in bad])