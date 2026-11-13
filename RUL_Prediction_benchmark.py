# -*- coding: utf-8 -*-
"""
PatchFormer 锂电池 RUL 复现：统一实验入口（与 RUL_Prediction_PatchFormer.py 逻辑完全一致，
只做三处必要的工程化改动，保证与原脚本结果可比）：
  1) --model 通过注册表分派到 PatchFormer / 三个消融变体 / 9 个基线模型；
  2) --start_point_list 改为逗号分隔字符串，结果 pth 按实际 SP 列表动态构造 key
     （原脚本把 SP300/400/500 写死，只跑部分起点时会 KeyError）；
  3) 每次实验额外追加一行机器可读指标到 results/metrics_<model>_<battery>.csv，
     并记录 best ckpt 路径，便于论文表格溯源与权重备份。

用法示例：
  python RUL_Prediction_benchmark.py --model PatchFormer --test_name CS2_35 --count 1 --max_epochs 100
  python RUL_Prediction_benchmark.py --model PatchTST     --test_name CS2_35 --count 1 --max_epochs 60
"""
import os
import csv
import time
import importlib

from assistant import get_gpus_memory_info

id, _ = get_gpus_memory_info()
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer
from pytorch_forecasting.metrics import SMAPE
import warnings

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use('agg')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='PatchFormer', help='Model name.')
parser.add_argument('--dataset', default='CALCE', choices=['CALCE', 'TJU'], help='Dataset name.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--root_dir', type=str, default='CALCE_RUL_prediction_sl_64', help='root path of the store file')
parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
parser.add_argument('--patch_len', type=int, default=2, help='patch length')
parser.add_argument('--d_model', type=int, default=16, help='hidden dimensions of model')
parser.add_argument('--n_heads', type=int, default=8, help='attention heads (PatchFormer family)')
parser.add_argument('--e_layers', type=int, default=2, help='encoder layers (PatchFormer family)')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--factor', type=int, default=3, help='attention factor (PatchFormer family)')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gradient_clip_val', type=float, default=0.2, help='gradient clipping value')
parser.add_argument('--tag', type=str, default='', help='suffix appended to output filenames, e.g. _tuned')
parser.add_argument('--count', type=int, default=10, help='The number of independent experiment.')
parser.add_argument('--batch_size', type=int, default=128, help='The batch size.')
parser.add_argument('--data_dir', type=str, default='data/CALCE data/', help='path of the data file')
parser.add_argument('--Battery_list', type=list, default=['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38'], help='Battery data')
parser.add_argument('--Rated_Capacity', type=float, default=1.1, help='Rate Capacity')
parser.add_argument('--test_name', type=str, default='CS2_35', help='Battery data used for test')
parser.add_argument('--start_point_list', type=str, default='300,400,500',
                    help='comma separated, e.g. 300,400,500')
parser.add_argument('--max_epochs', type=int, default=100, help='max train epochs (main model default; baselines use 60, ablations 80)')
args = parser.parse_args()
args.start_point_list = [int(s) for s in str(args.start_point_list).split(',') if s.strip() != '']

# ----------------------------------------- 模型注册表 ----------------------------------------------
# 全部为 ModelsModify 下已有的 BaseModel 封装（与官方脚本同一套 from_dataset 接口）
MODEL_REGISTRY = {
    # 本文方法
    'PatchFormer': ('ModelsModify.PatchFormer', 'PatchFormerNetModel'),
    # 三个消融变体
    'PatchFormer_no_DPAN': ('ModelsModify.PatchFormer_no_DPAN', 'PatchFormer_no_DPANNetModel'),
    'PatchFormer_no_inter_PA': ('ModelsModify.PatchFormer_no_inter_PA', 'PatchFormer_no_inter_PANetModel'),
    'PatchFormer_no_intra_PA': ('ModelsModify.PatchFormer_no_intra_PA', 'PatchFormer_no_intra_PANetModel'),
    # 基线
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
    if name not in MODEL_REGISTRY:
        raise ValueError('未知模型: {}，可选: {}'.format(name, list(MODEL_REGISTRY.keys())))
    module_name, class_name = MODEL_REGISTRY[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_model(name, training):
    """按模型给出与官方默认一致的构造参数（PatchFormer 家族共享一套超参，保证消融公平）。"""
    cls = import_model_class(name)
    enc_in = 1  # 输入特征只有 Capacity（encoder_cont[:, :, :-1]）
    common = dict(learning_rate=args.learning_rate, loss=SMAPE())
    patch_family = dict(patch_len=args.patch_len, seq_len=args.seq_len, pred_len=args.pred_len,
                        enc_in=enc_in, d_model=args.d_model, n_heads=args.n_heads,
                        e_layers=args.e_layers, factor=args.factor, dropout=args.dropout, **common)
    if name in ('PatchFormer', 'PatchFormer_no_DPAN', 'PatchFormer_no_inter_PA', 'PatchFormer_no_intra_PA'):
        kwargs = patch_family
    elif name == 'PatchTST':
        kwargs = dict(patch_len=args.patch_len, stride=args.patch_len, seq_len=args.seq_len,
                      pred_len=args.pred_len, enc_in=enc_in, d_model=args.d_model, **common)
    elif name == 'iTransformer':
        kwargs = dict(input_size=args.seq_len, output_size=args.pred_len, d_model=args.d_model,
                      enc_in=enc_in, n_heads=args.n_heads, e_layers=args.e_layers,
                      d_ff=2 * args.d_model, dropout=args.dropout, **common)
    elif name == 'ModernTCN':
        kwargs = dict(enc_in=enc_in, input_size=args.seq_len, output_size=args.pred_len,
                      hidden_size=64, P=4, S=2, kernel_sizes=[5, 51], r=2, num_layers=2, **common)
    elif name == 'TimesNet':
        kwargs = dict(seq_len=args.seq_len, pred_len=args.pred_len, enc_in=enc_in, dec_in=enc_in,
                      c_out=1, d_model=args.d_model, e_layers=args.e_layers,
                      d_ff=2 * args.d_model, dropout=args.dropout, **common)
    elif name == 'TimeMixer':
        kwargs = dict(seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
                      enc_in=enc_in, c_out=1, d_model=args.d_model, e_layers=args.e_layers,
                      d_ff=2 * args.d_model, dropout=args.dropout, **common)
    elif name == 'MambaSimple':
        kwargs = dict(seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
                      enc_in=enc_in, dec_in=enc_in, c_out=1, d_model=args.d_model,
                      e_layers=args.e_layers, d_ff=2 * args.d_model, dropout=args.dropout, **common)
    elif name in ('Autoformer', 'FEDformer'):
        kwargs = dict(seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
                      enc_in=enc_in, dec_in=enc_in, c_out=1, d_model=args.d_model,
                      e_layers=args.e_layers, d_ff=2 * args.d_model, dropout=args.dropout, **common)
    elif name == 'PathFormer':
        kwargs = dict(enc_in=enc_in, seq_len=args.seq_len, pred_len=args.pred_len,
                      k=3, patch_size_list=[2, 4, 8, 16], **common)
    else:
        raise ValueError('未实现构造参数: {}'.format(name))
    return cls.from_dataset(training, **kwargs)


# =================================== 以下与官方 RUL_Prediction_PatchFormer.py 完全一致 ===============================

def rul_value_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0

    for i in range(len(y_test) - 1):
        if y_test[i] <= threshold >= y_test[i + 1]:
            true_re = i - 1
            break

    for i in range(len(y_predict) - 1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break

    rul_real = true_re + 1
    rul_pred = pred_re + 1
    ae_error = abs(true_re - pred_re)
    re_score = abs(true_re - pred_re) / true_re
    if re_score > 1:
        re_score = 1

    return rul_real, rul_pred, ae_error, re_score


# 打印日志到文件
def _test_id(name):
    """从电池名中提取数字编号（CS2_35 -> 35，CY25_1 -> 1）。"""
    digits = ''.join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else 0


def print_log(print_string, log_file, visible=True):
    if visible:
        print("{}".format(print_string))
    log_file.write('{}\n'.format(print_string))
    log_file.flush()


if args.dataset == 'TJU':
    from TJUDataPreProcess import BatteryDataProcess
    _npy_name = 'Dataset_3_NCM_NCA_battery_1C.npy'
else:
    from CALCEDataPreProcess import BatteryDataProcess
    _npy_name = 'CALCE_Data.npy'

# ------------------------------------------------- step 1: 数据准备 ----------------------------------------
BatteryData = np.load(os.path.join(args.data_dir, _npy_name), allow_pickle=True)
BatteryData = BatteryData.item()

_, _, df_all = BatteryDataProcess(BatteryData, args.test_name, args.start_point_list[0], args)
real_data = df_all['target'].values * args.Rated_Capacity
all_pred_data_list = []

root_dir = 'results_{}/{}/{}/'.format(args.root_dir, args.test_name, args.model)
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

metrics_csv = 'results/metrics_{}_{}{}.csv'.format(args.model, args.test_name, args.tag)
os.makedirs('results', exist_ok=True)
write_header = not os.path.exists(metrics_csv)

for start_point in args.start_point_list:
    df_train, df_test, df_all = BatteryDataProcess(BatteryData, args.test_name, start_point, args)
    mask_len = len(df_train)
    time_varying_known_reals = ['Capacity']
    time_varying_unknown_reals = ['target']
    max_prediction_length = args.pred_len
    max_encoder_length = args.seq_len

    training = TimeSeriesDataSet(
        df_train[0:int(0.8 * mask_len)],
        time_idx="time_idx",
        target="target",
        group_ids=['group_id'],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=EncoderNormalizer(),
        add_encoder_length=False
    )
    train_dataloader = training.to_dataloader(train=True, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                              drop_last=True)
    for x, (y, weight) in iter(train_dataloader):
        print("['encoder_cont']:", x['encoder_cont'].shape)
        print("['decoder_cont']:", x['decoder_cont'].shape)
        en_feats_num = x['encoder_cat'].shape[-1] + x['encoder_cont'].shape[-1]
        de_feats_num = x['decoder_cat'].shape[-1] + x['decoder_cont'].shape[-1]
        print('y:', y[0].shape)
        break
    validing = TimeSeriesDataSet(
        df_train[int(0.8 * mask_len):],
        time_idx="time_idx",
        target='target',
        group_ids=['group_id'],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=EncoderNormalizer(),
        add_encoder_length=False
    )
    val_dataloader = validing.to_dataloader(train=False, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                            drop_last=False)
    testing = TimeSeriesDataSet(
        df_test,
        time_idx="time_idx",
        target='target',
        group_ids=['group_id'],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=EncoderNormalizer(),
        add_encoder_length=False
    )
    test_dataloader = testing.to_dataloader(train=False, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                            drop_last=False)

    sp_root_dir = os.path.join(root_dir, 'SP{}/'.format(start_point))
    if not os.path.exists(sp_root_dir):
        os.makedirs(sp_root_dir)

    after_name = 'in_l_{}_out_l_{}_Pcap'.format(max_encoder_length, max_prediction_length)
    stat_log_path = os.path.join(sp_root_dir,
                                 'log_stat_Feas_{}_{}_{}.txt'.format(len(training.reals), en_feats_num, after_name))
    stat_log = open(stat_log_path, 'w', encoding='UTF-8')
    print_log('model name:{}\n'.format(args.model), stat_log)
    print_log('selected battery name:{}, start point:{}\n'.format(args.test_name, start_point), stat_log)

    MAE_avg = 0.
    RMSE_avg = 0.
    r2_avg = 0.
    RE_avg = 0.
    AE_avg = 0.
    RUL_real_avg = 0.
    RUL_pred_avg = 0.
    epoch_avg = 0.
    count = 0
    stat_pred_data_list = []
    from assistant import set_seed

    while count < args.count:
        count += 1
        args.seed = set_seed(count)
        # --------------------------------------step 2: 网络构建--------------------------------------------------------
        model = build_model(args.model, training)
        after_name = 'in_l_{}_out_l_{}_Pcap'.format(max_encoder_length, max_prediction_length)

        save_dir = sp_root_dir + '{}/'.format(str(model).split('\n')[0][:-1] + time.strftime(
            '_%Y-%m-%d %X', time.localtime()).replace(':', '-'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        log_path = os.path.join(save_dir,
                                'log_Feas_{}_{}_{}.txt'.format(len(training.reals), en_feats_num, after_name))
        log = open(log_path, 'w', encoding='UTF-8')

        study_model_path = os.path.join(save_dir, str(model).split('\n')[0][:-1])
        if not os.path.exists(study_model_path):
            os.makedirs(study_model_path)

        print_log(('train dataset:{},val:{},test:{}'.format(int(0.8 * mask_len), int(0.1 * mask_len), len(df_test))),
                  log)
        print_log(('Input Feature num:{} ,name:{}'.format(len(training.reals), training.reals)), log)
        print_log('model name:{}\n'.format(args.model), log)
        print_log((f"Number of parameters in network(参数数量(Params): {model.size() / 1e3:.1f}k"), log)

        # --------------------------------------step 3: 训练--------------------------------------------------------
        early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=False,
                                            mode='min')

        use_gpu = torch.cuda.is_available()
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            gpus=1 if use_gpu else 0,
            accelerator='gpu' if use_gpu else 'cpu',
            gradient_clip_val=args.gradient_clip_val,
            callbacks=[early_stop_callback],
            logger=False,
            default_root_dir=study_model_path,
        )
        t0 = time.time()
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        train_seconds = time.time() - t0

        # *** 加载最优模型(本循环内) ***
        best_model_path = trainer.checkpoint_callback.best_model_path
        print('best_model_path:', best_model_path)
        print(('best_model_path:{}'.format(best_model_path)), log)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_class = import_model_class(args.model)
        best_model = model_class.load_from_checkpoint(best_model_path).to(device=device)

        # 预测方式,不实时更新
        predictions = best_model.predict(test_dataloader, batch_size=256)
        predictions = predictions.detach().cpu().numpy().reshape(-1)
        actuals_df = df_all.loc[df_all['Cycle'] >= start_point, ['Cycle', 'target']]
        actuals = actuals_df['target'].values
        y_true = actuals * args.Rated_Capacity
        y_pred = predictions * args.Rated_Capacity

        stat_pred_data_list.append(y_pred)
        mask = y_true >= 0.
        from sklearn.metrics import r2_score
        MAE = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        RMSE = np.sqrt(np.mean(np.square(y_true[mask] - y_pred[mask])))
        r2 = r2_score(y_true[mask], y_pred[mask])
        RUL_real, RUL_pred, AE, RE = rul_value_error(y_true[mask], y_pred[mask], threshold=args.Rated_Capacity * 0.7)
        epoch_avg += trainer.current_epoch
        print_log("实际运行的Epoch数量: {}  \n".format(trainer.current_epoch), log)
        print_log((
            '第{}次实验的结果: \nMAE:{:.4f}, RMSE:{:.4f}, r2:{:.4f}, RUL_real:{}, RUL_pred:{}, AE:{}, RE:{:.4f}  \n'.format(
                count, MAE, RMSE, r2, RUL_real, RUL_pred, AE, RE)), log)
        print_log("第{}次实验实际运行的Epoch数量: {}  \n".format(count, trainer.current_epoch), stat_log)
        print_log((
            '第{}次实验的结果: \nMAE:{:.4f}, RMSE:{:.4f}, r2:{:.4f}, RUL_real:{}, RUL_pred:{}, AE:{}, RE:{:.4f}  \n'.format(
                count, MAE, RMSE, r2, RUL_real, RUL_pred, AE, RE)), stat_log)

        # 机器可读指标（论文表格的唯一数据源）
        with open(metrics_csv, 'a', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            if write_header:
                writer.writerow(['model', 'test_name', 'sp', 'run', 'epochs', 'train_seconds',
                                 'MAE', 'RMSE', 'r2', 'RUL_real', 'RUL_pred', 'AE', 'RE', 'best_model_path'])
                write_header = False
            writer.writerow([args.model, args.test_name, start_point, count, trainer.current_epoch,
                             round(train_seconds, 2), round(float(MAE), 6), round(float(RMSE), 6),
                             round(float(r2), 6), int(RUL_real), int(RUL_pred), int(AE), round(float(RE), 6),
                             best_model_path])

        MAE_avg += MAE
        RMSE_avg += RMSE
        r2_avg += r2
        RE_avg += RE
        AE_avg += AE
        RUL_real_avg += RUL_real
        RUL_pred_avg += RUL_pred

        del model
        log.close()

    all_pred_data_list.append(stat_pred_data_list)
    MAE_avg /= args.count
    RMSE_avg /= args.count
    r2_avg /= args.count
    RE_avg /= args.count
    AE_avg /= args.count
    RUL_real_avg /= args.count
    RUL_pred_avg /= args.count
    epoch_avg /= args.count
    print_log("实际运行的Epoch数量的平均值: {}  \n".format(epoch_avg), stat_log)
    print_log((
        '{}次实验的统计平均值: \nMAE:{:.4f}, RMSE:{:.4f}, r2:{:.4f}, RUL_real:{}, RUL_pred:{}, AE:{}, RE:{:.4f}  \n'.format(
            args.count, MAE_avg, RMSE_avg, r2_avg, RUL_real_avg, RUL_pred_avg, AE_avg, RE_avg)), stat_log)
    setting = '{}_sl{}_ll{}_pl{}_bs{}_ct{}_tn{}_me{}'.format(
        args.model,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.batch_size,
        args.count,
        _test_id(args.test_name),
        args.max_epochs)
    print_log('args setting : {}\n'.format(setting), stat_log)
    stat_log.close()

# 结果按实际 SP 列表存 dict（修复原脚本写死 SP300/400/500 的问题）
results = dict()
for i, sp in enumerate(args.start_point_list):
    results['SP{}'.format(sp)] = all_pred_data_list[i]
pth_path = 'results/RUL_{}_{}{}.pth'.format(args.test_name, args.model, args.tag)
torch.save(results, pth_path)
print('saved: {}'.format(pth_path))