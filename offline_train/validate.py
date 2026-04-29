"""
validate.py
使用验证集 CSV 评估已训练的 Predictor 效果

使用方法：
  cd /apollo/modules/MiningTruckTrack
  python3 -m offline_train.validate --csv /path/to/val/*.csv --config config_10s_h30_lstm.yaml --model state_dict/predictor_final.pth
"""

import os
import sys
import csv
import argparse
import math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_model.parser import parse_config
from network_model.predictor import Predictor
from offline_train.data_process import (
    process_row, build_sequences,
    LAT_MIN, LAT_MAX, S_MIN, S_MAX,
    HEAD_MIN, HEAD_MAX, V_MIN, V_MAX,
    A_MIN, A_MAX, KAPPA_MIN, KAPPA_MAX,
    HORIZON, TRAJ_POINTS,
)


# 反归一化：norm ∈ [-1,1] → real
def denormalize(x, xmin, xmax):
    return (x + 1.0) * (xmax - xmin) / 2.0 + xmin


# 各维度反归一化参数
DENORM = [
    (LAT_MIN,   LAT_MAX,   'm',     'lat'),
    (S_MIN,     S_MAX,     'm',     's'),
    (HEAD_MIN,  HEAD_MAX,  'rad',   'head'),
    (V_MIN,     V_MAX,     'm/s',   'v'),
    (A_MIN,     A_MAX,     'm/s²',  'a'),
    (KAPPA_MIN, KAPPA_MAX, '1/m',   'kappa'),
]


def load_csv(csv_path):
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                rows.append(row)
    return rows


def process_csv_files(csv_paths, csv_traj_points=20):
    all_states, all_targets, all_contexts = [], [], []
    total_skip = 0
    for path in csv_paths:
        rows = load_csv(path)
        ok = 0
        for row in rows:
            result = process_row(row, csv_traj_points)
            if result is None:
                total_skip += 1
                continue
            s, t, c = result
            all_states.append(s)
            all_targets.append(t)
            all_contexts.append(c)
            ok += 1
        print(f'  {os.path.basename(path)}: {ok} 条轨迹')
    print(f'共 {len(all_states)} 条轨迹，跳过 {total_skip} 条')
    return all_states, all_targets, all_contexts


def rollout(predictor, state_seq, target_seq, context_seq, hist_horizon, rollout_steps):
    """
    对单条样本做闭环预测
    state_seq:  (2H+1, 6)
    target_seq: (3H,   3)
    context_seq:(2,)
    返回 pred_states: (rollout_steps, 6) — 归一化空间
    """
    device = predictor.device
    H = hist_horizon

    # 当前帧为中心点
    state          = torch.tensor(state_seq[H],    dtype=torch.float32).unsqueeze(0).to(device)
    previous_state = torch.tensor(state_seq[:H],   dtype=torch.float32).unsqueeze(0).to(device)
    target         = torch.tensor(target_seq[H],   dtype=torch.float32).unsqueeze(0).to(device)
    previous_target= torch.tensor(target_seq[:H],  dtype=torch.float32).unsqueeze(0).to(device)
    context        = torch.tensor(context_seq,     dtype=torch.float32).unsqueeze(0).to(device)

    pred_states = []
    with torch.no_grad():
        for k in range(rollout_steps):
            next_target = torch.tensor(
                target_seq[H + k + 1], dtype=torch.float32).unsqueeze(0).to(device)
            next_state, _, _hx = predictor.predict(
                state, previous_state, target, previous_target, next_target, context)

            pred_states.append(next_state.squeeze(0).cpu().numpy())

            # 滚动窗口
            previous_state = torch.cat([previous_state[:, 1:, :], state.unsqueeze(1)], dim=1)
            previous_target = torch.cat([previous_target[:, 1:, :], target.unsqueeze(1)], dim=1)
            state  = next_state
            target = next_target

    return np.array(pred_states)  # (rollout_steps, 6)


def validate(args):
    root = os.path.dirname(os.path.dirname(__file__))
    cfg = parse_config(os.path.join(root, args.config))

    print(f'\n加载模型: {args.model}')
    predictor = Predictor(cfg)
    predictor.load(args.model)
    predictor.eval()

    print(f'\n读取验证 CSV...')
    csv_traj_points = getattr(cfg, 'traj_points', 20)
    csv_paths = args.csv
    all_states, all_targets, all_contexts = process_csv_files(csv_paths, csv_traj_points)

    print(f'\n构建验证样本...')
    stride = getattr(args, 'stride', 1)
    state_seqs, target_seqs, _, context_seqs = build_sequences(
        all_states, all_targets, all_contexts, cfg.horizon,
        traj_points=csv_traj_points, stride=stride)
    N = len(state_seqs)
    print(f'共 {N} 个验证样本')

    # ── 逐样本预测 ─────────────────────────────────────────────────────────
    H = cfg.horizon
    all_mae_norm = np.zeros((N, cfg.train_horizon, cfg.state_dim), dtype=np.float32)

    for i in range(N):
        pred = rollout(predictor, state_seqs[i], target_seqs[i], context_seqs[i],
                       cfg.horizon, cfg.train_horizon)
        real = state_seqs[i, H + 1: H + 1 + cfg.train_horizon, :]  # (train_horizon, 6)
        all_mae_norm[i] = np.abs(pred - real)

    # ── 按步统计 MAE ────────────────────────────────────────────────────────
    # mean over samples: (train_horizon, state_dim)
    mae_per_step = all_mae_norm.mean(axis=0)
    mae_avg      = all_mae_norm.mean(axis=(0, 1))  # (state_dim,)

    print(f'\n{"="*70}')
    print(f'验证结果（{N} 个样本，{cfg.train_horizon} 步预测，每步 0.1s）')
    print(f'{"="*70}')

    # 表头
    dim_names = [d[3] for d in DENORM]
    dim_units = [d[2] for d in DENORM]
    header = f'{"step":>5}' + ''.join(f'{n:>12}' for n in dim_names)
    print(header)
    print('-' * len(header))

    for step in range(cfg.train_horizon):
        t = (step + 1) * 0.1
        row_parts = [f'{t:.1f}s']
        for dim_i, (xmin, xmax, unit, name) in enumerate(DENORM):
            mae_real = mae_per_step[step, dim_i] * (xmax - xmin) / 2.0
            row_parts.append(f'{mae_real:.4f}{unit}')
        print(f'{"":>5}' + ''.join(f'{p:>12}' for p in row_parts))

    print('-' * len(header))
    avg_parts = ['avg']
    for dim_i, (xmin, xmax, unit, name) in enumerate(DENORM):
        mae_real = mae_avg[dim_i] * (xmax - xmin) / 2.0
        avg_parts.append(f'{mae_real:.4f}{unit}')
    print(f'{"":>5}' + ''.join(f'{p:>12}' for p in avg_parts))
    print(f'{"="*70}')

    # ── 保存结果 ────────────────────────────────────────────────────────────
    if args.save:
        out_path = args.save
        with open(out_path, 'w') as f:
            f.write('step,' + ','.join(dim_names) + '\n')
            for step in range(cfg.train_horizon):
                t = (step + 1) * 0.1
                vals = [f'{t:.1f}']
                for dim_i, (xmin, xmax, unit, _) in enumerate(DENORM):
                    mae_real = mae_per_step[step, dim_i] * (xmax - xmin) / 2.0
                    vals.append(f'{mae_real:.6f}')
                f.write(','.join(vals) + '\n')
        print(f'\n结果已保存到 {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',   nargs='+', required=True, help='验证集 CSV 文件')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='配置文件路径，相对于项目根目录，如 config_10s_h30_lstm.yaml')
    parser.add_argument('--model', required=True,            help='predictor 权重路径')
    parser.add_argument('--save',  type=str, default=None,   help='将逐步 MAE 保存为 CSV')
    parser.add_argument('--stride', type=int, default=1,
                        help='验证样本滑窗步长，>1 可减少样本数（默认 1）')
    args = parser.parse_args()
    validate(args)
