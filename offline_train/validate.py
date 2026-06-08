"""
validate.py
使用验证集 CSV 评估已训练的 Predictor（及可选 Controller）效果

使用方法：
  cd /apollo/modules/MiningTruckTrack
  # 只验证 predictor
  python3 -m offline_train.validate --csv /path/to/val/*.csv --config config_10s_h30_lstm.yaml --model predictor.pth
  # 同时验证 controller
  python3 -m offline_train.validate --csv /path/to/val/*.csv --config config_10s_h30_lstm.yaml --model predictor.pth --controller controller.pth
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_model.parser import parse_config
from network_model.predictor import Predictor
from network_model.controller import Controller
from offline_train.data_process import (
    process_row, build_sequences,
    LAT_MIN, LAT_MAX, S_MIN, S_MAX,
    HEAD_MIN, HEAD_MAX, V_MIN, V_MAX,
    A_MIN, A_MAX, KAPPA_MIN, KAPPA_MAX,
    HORIZON, TRAJ_POINTS,
)


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
        next(reader, None)
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
    无 controller 的闭环预测（predictor 直接用 real target）
    返回 pred_states: (rollout_steps, 6) — 归一化空间
    """
    device = predictor.device
    H = hist_horizon

    state           = torch.tensor(state_seq[H],   dtype=torch.float32).unsqueeze(0).to(device)
    previous_state  = torch.tensor(state_seq[:H],  dtype=torch.float32).unsqueeze(0).to(device)
    target          = torch.tensor(target_seq[H],  dtype=torch.float32).unsqueeze(0).to(device)
    previous_target = torch.tensor(target_seq[:H], dtype=torch.float32).unsqueeze(0).to(device)
    context         = torch.tensor(context_seq,    dtype=torch.float32).unsqueeze(0).to(device)

    pred_states = []
    with torch.no_grad():
        for k in range(rollout_steps):
            next_target = torch.tensor(
                target_seq[H + k + 1], dtype=torch.float32).unsqueeze(0).to(device)
            next_state, _, _ = predictor.predict(
                state, previous_state, target, previous_target, next_target, context)

            pred_states.append(next_state.squeeze(0).cpu().numpy())

            previous_state  = torch.cat([previous_state[:, 1:, :], state.unsqueeze(1)], dim=1)
            previous_target = torch.cat([previous_target[:, 1:, :], target.unsqueeze(1)], dim=1)
            state  = next_state
            target = next_target

    return np.array(pred_states)  # (rollout_steps, 6)


def rollout_with_controller(predictor, controller, state_seq, target_seq, context_seq,
                             hist_horizon, rollout_steps):
    """
    有 controller 的闭环预测：每步用 controller 调整 next_target 后再交给 predictor
    返回 pred_states: (rollout_steps, 6) — 归一化空间
    """
    device = predictor.device
    H = hist_horizon

    state           = torch.tensor(state_seq[H],   dtype=torch.float32).unsqueeze(0).to(device)
    previous_state  = torch.tensor(state_seq[:H],  dtype=torch.float32).unsqueeze(0).to(device)
    target          = torch.tensor(target_seq[H],  dtype=torch.float32).unsqueeze(0).to(device)
    previous_target = torch.tensor(target_seq[:H], dtype=torch.float32).unsqueeze(0).to(device)
    context         = torch.tensor(context_seq,    dtype=torch.float32).unsqueeze(0).to(device)

    pred_states = []
    with torch.no_grad():
        for k in range(rollout_steps):
            # 未来 H 步 real target（超出末尾时用最后一步填充）
            fut_indices = [min(H + k + 1 + h, len(target_seq) - 1) for h in range(H)]
            future_real_target = torch.tensor(
                np.array([target_seq[idx] for idx in fut_indices], dtype=np.float32),
                dtype=torch.float32).unsqueeze(0).to(device)  # (1, H, 3)

            next_target, _ = controller.control(
                state, previous_state, target, previous_target, future_real_target, context)
            next_target = torch.clamp(next_target, -1.0, 1.0)

            next_state, _, _ = predictor.predict(
                state, previous_state, target, previous_target, next_target, context)

            pred_states.append(next_state.squeeze(0).cpu().numpy())

            previous_state  = torch.cat([previous_state[:, 1:, :], state.unsqueeze(1)], dim=1)
            previous_target = torch.cat([previous_target[:, 1:, :], target.unsqueeze(1)], dim=1)
            state  = next_state
            target = next_target

    return np.array(pred_states)  # (rollout_steps, 6)


def _print_table(label, N, train_horizon, mae_per_step, mae_avg):
    dim_names = [d[3] for d in DENORM]
    print(f'\n{"="*76}')
    print(f'{label}（{N} 个样本，{train_horizon} 步，每步 0.1s）')
    print(f'{"="*76}')
    header = f'{"step":>6}' + ''.join(f'{n:>11}' for n in dim_names)
    print(header)
    print('-' * len(header))
    for step in range(train_horizon):
        t = (step + 1) * 0.1
        row = [f'{t:.1f}s']
        for dim_i, (xmin, xmax, unit, _) in enumerate(DENORM):
            row.append(f'{mae_per_step[step, dim_i] * (xmax - xmin) / 2.0:.4f}{unit}')
        print(f'{"":>6}' + ''.join(f'{p:>11}' for p in row))
    print('-' * len(header))
    avg = ['avg']
    for dim_i, (xmin, xmax, unit, _) in enumerate(DENORM):
        avg.append(f'{mae_avg[dim_i] * (xmax - xmin) / 2.0:.4f}{unit}')
    print(f'{"":>6}' + ''.join(f'{p:>11}' for p in avg))
    print(f'{"="*76}')


def build_ideal_refs(target_seq, H, rollout_steps, state_dim):
    """
    构造理想参考状态（normalized空间）：
    [lat_err=0, s_err=0, head_err=0, v=v_ref, a=a_ref, kappa=kappa_ref]
    target_seq: (T, 3) normalized，每行 [v_ref, a_ref, kappa_ref]
    """
    def norm_zero(xmin, xmax):
        return (0.0 - xmin) / (xmax - xmin) * 2.0 - 1.0

    lat_zero  = norm_zero(LAT_MIN,  LAT_MAX)
    s_zero    = norm_zero(S_MIN,    S_MAX)
    head_zero = norm_zero(HEAD_MIN, HEAD_MAX)

    ideal = np.zeros((rollout_steps, state_dim), dtype=np.float32)
    for k in range(rollout_steps):
        t_idx = min(H + k + 1, len(target_seq) - 1)
        ideal[k, 0] = lat_zero
        ideal[k, 1] = s_zero
        ideal[k, 2] = head_zero
        ideal[k, 3] = target_seq[t_idx, 0]  # v_ref
        ideal[k, 4] = target_seq[t_idx, 1]  # a_ref
        ideal[k, 5] = target_seq[t_idx, 2]  # kappa_ref
    return ideal


def validate(args):
    root = os.path.dirname(os.path.dirname(__file__))
    cfg = parse_config(os.path.join(root, args.config))

    print(f'\n加载 Predictor: {args.model}')
    predictor = Predictor(cfg)
    predictor.load(args.model)
    predictor.eval()

    controller = None
    if args.controller:
        print(f'加载 Controller: {args.controller}')
        controller = Controller(cfg)
        controller.load(args.controller)
        controller.eval()

        if args.diag:
            print('\n=== Controller 输出诊断（需先读取数据）===')
            diag_csv = args.csv[:1]
            diag_traj = getattr(cfg, 'traj_points', 20)
            diag_states, diag_targets, diag_contexts = process_csv_files(diag_csv, diag_traj)
            diag_state_seqs, diag_target_seqs, _, diag_context_seqs = build_sequences(
                diag_states, diag_targets, diag_contexts, cfg.horizon,
                traj_points=diag_traj, stride=10)
            B = min(512, len(diag_state_seqs))
            H = cfg.horizon
            device = torch.device(cfg.device)
            idx = np.random.choice(len(diag_state_seqs), B, replace=False)
            s   = torch.tensor(diag_state_seqs[idx, H, :],    dtype=torch.float32).to(device)
            ps  = torch.tensor(diag_state_seqs[idx, :H, :],   dtype=torch.float32).to(device)
            t   = torch.tensor(diag_target_seqs[idx, H, :],   dtype=torch.float32).to(device)
            pt  = torch.tensor(diag_target_seqs[idx, :H, :],  dtype=torch.float32).to(device)
            ctx = torch.tensor(diag_context_seqs[idx],        dtype=torch.float32).to(device)
            fut_idx = np.clip(np.arange(1, H+1) + H, 0, diag_target_seqs.shape[1]-1)
            fut = torch.tensor(diag_target_seqs[np.ix_(idx, fut_idx)],
                               dtype=torch.float32).to(device)
            with torch.no_grad():
                x = torch.cat([ps.reshape(B,-1), s, pt.reshape(B,-1), t,
                                fut.reshape(B,-1), ctx], dim=-1)
                raw = controller._controller(x)
                _, delta = controller.control(s, ps, t, pt, fut, ctx)
            print(f'  样本数: {B}，来自真实数据')
            print(f'  raw (tanh输出): mean={raw.mean(dim=0).cpu().numpy().round(4)}, '
                  f'std={raw.std(dim=0).cpu().numpy().round(4)}')
            print(f'  delta(×max_range): mean={delta.mean(dim=0).cpu().numpy().round(4)}, '
                  f'std={delta.std(dim=0).cpu().numpy().round(4)}')
            return

    print(f'\n读取验证 CSV...')
    csv_traj_points = getattr(cfg, 'traj_points', 20)
    all_states, all_targets, all_contexts = process_csv_files(args.csv, csv_traj_points)

    print(f'\n构建验证样本...')
    stride = getattr(args, 'stride', 1)
    state_seqs, target_seqs, _, context_seqs = build_sequences(
        all_states, all_targets, all_contexts, cfg.horizon,
        traj_points=csv_traj_points, stride=stride)
    N = len(state_seqs)
    print(f'共 {N} 个验证样本')
    if N == 0:
        print('无有效样本，退出')
        return

    H = cfg.horizon
    all_mae_norm  = np.zeros((N, cfg.train_horizon, cfg.state_dim), dtype=np.float32)
    all_mae_ctrl  = np.zeros((N, cfg.train_horizon, cfg.state_dim), dtype=np.float32) \
                    if controller else None

    print('\n开始评估...')
    for i in range(N):
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{N}')
        ideal = build_ideal_refs(target_seqs[i], H, cfg.train_horizon, cfg.state_dim)

        pred = rollout(predictor, state_seqs[i], target_seqs[i], context_seqs[i],
                       H, cfg.train_horizon)
        all_mae_norm[i] = np.abs(pred - ideal)

        if controller is not None:
            pred_c = rollout_with_controller(
                predictor, controller, state_seqs[i], target_seqs[i], context_seqs[i],
                H, cfg.train_horizon)
            all_mae_ctrl[i] = np.abs(pred_c - ideal)

    mae_per_step = all_mae_norm.mean(axis=0)
    mae_avg      = all_mae_norm.mean(axis=(0, 1))

    _print_table('无 Controller（开环预测）', N, cfg.train_horizon, mae_per_step, mae_avg)

    if controller is not None:
        mae_per_step_c = all_mae_ctrl.mean(axis=0)
        mae_avg_c      = all_mae_ctrl.mean(axis=(0, 1))
        _print_table('有 Controller（闭环预测）', N, cfg.train_horizon, mae_per_step_c, mae_avg_c)

        # 对比摘要
        dim_names = [d[3] for d in DENORM]
        print(f'\n{"="*76}')
        print('对比摘要：有 Controller vs 无 Controller（MAE 变化，负数=改善）')
        print(f'{"="*76}')
        header = f'{"step":>6}' + ''.join(f'{n:>11}' for n in dim_names)
        print(header)
        print('-' * len(header))
        for step in range(cfg.train_horizon):
            t = (step + 1) * 0.1
            row = [f'{t:.1f}s']
            for dim_i, (xmin, xmax, unit, _) in enumerate(DENORM):
                scale = (xmax - xmin) / 2.0
                diff = (mae_per_step_c[step, dim_i] - mae_per_step[step, dim_i]) * scale
                row.append(f'{diff:+.4f}{unit}')
            print(f'{"":>6}' + ''.join(f'{p:>11}' for p in row))
        print(f'{"="*76}')

    # 保存 CSV
    if args.save:
        dim_names = [d[3] for d in DENORM]
        with open(args.save, 'w') as f:
            cols = ['step'] + dim_names
            if controller is not None:
                cols += [n + '_ctrl' for n in dim_names]
            f.write(','.join(cols) + '\n')
            for step in range(cfg.train_horizon):
                t = (step + 1) * 0.1
                vals = [f'{t:.1f}']
                for dim_i, (xmin, xmax, _, __) in enumerate(DENORM):
                    vals.append(f'{mae_per_step[step, dim_i] * (xmax - xmin) / 2.0:.6f}')
                if controller is not None:
                    for dim_i, (xmin, xmax, _, __) in enumerate(DENORM):
                        vals.append(f'{mae_per_step_c[step, dim_i] * (xmax - xmin) / 2.0:.6f}')
                f.write(','.join(vals) + '\n')
        print(f'\n结果已保存到 {args.save}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        nargs='+', required=True, help='验证集 CSV 文件')
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='配置文件路径，相对于项目根目录')
    parser.add_argument('--model',      required=True,           help='predictor 权重路径')
    parser.add_argument('--controller', type=str, default=None,  help='controller 权重路径（可选）')
    parser.add_argument('--save',       type=str, default=None,  help='将逐步 MAE 保存为 CSV')
    parser.add_argument('--stride',     type=int, default=1,
                        help='验证样本滑窗步长，>1 可减少样本数（默认 1）')
    parser.add_argument('--diag',       action='store_true',
                        help='仅诊断 controller 输出分布，不跑完整验证')
    args = parser.parse_args()
    validate(args)
