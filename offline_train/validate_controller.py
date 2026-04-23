"""
validate_controller.py
对比有无 Controller 时的轨迹跟踪误差，评估 Controller 效果。

使用方法：
  # 1. 先用 data_process.py 处理验证集 CSV
  cd /apollo/modules/MiningTruckTrack
  python3 -m offline_train.data_process \
      --csv /path/to/val/*.csv \
      --outdir data/val/

  # 2. 再运行验证
  python3 -m offline_train.validate_controller \
      --data data/val/ \
      --predictor state_dict/predictor_final.pth \
      --controller state_dict/controller_final.pth

  # 同时保存逐步 MAE 对比
  python3 -m offline_train.validate_controller \
      --data data/val/ \
      --predictor state_dict/predictor_final.pth \
      --controller state_dict/controller_final.pth \
      --save logs/val_controller.csv
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_model.parser import parse_config
from network_model.predictor import Predictor
from network_model.controller import Controller
from offline_train.data_process import (
    V_MIN, V_MAX,
    A_MIN, A_MAX,
    KAPPA_MIN, KAPPA_MAX,
)

DENORM = [
    (V_MIN,     V_MAX,     'm/s',  'v'),
    (A_MIN,     A_MAX,     'm/s²', 'a'),
    (KAPPA_MIN, KAPPA_MAX, '1/m',  'kappa'),
]

# state 中 v/a/kappa 对应的维度索引
STATE_VAK_DIMS = [3, 4, 5]


def load_pt_data(data_dir):
    """从 data_process.py 生成的 .pt 文件加载数据"""
    state_seqs   = torch.load(os.path.join(data_dir, 'state.pt'))
    target_seqs  = torch.load(os.path.join(data_dir, 'target.pt'))
    context_seqs = torch.load(os.path.join(data_dir, 'context.pt'))
    return state_seqs, target_seqs, context_seqs


def rollout_baseline(predictor, state_seq, target_seq, context_seq, horizon):
    """不使用 Controller，直接用 real_target 滚动预测"""
    device = predictor.device
    H = horizon

    state           = torch.tensor(state_seq[H],   dtype=torch.float32).unsqueeze(0).to(device)
    previous_state  = torch.tensor(state_seq[:H],  dtype=torch.float32).unsqueeze(0).to(device)
    target          = torch.tensor(target_seq[H],  dtype=torch.float32).unsqueeze(0).to(device)
    previous_target = torch.tensor(target_seq[:H], dtype=torch.float32).unsqueeze(0).to(device)
    context         = torch.tensor(context_seq,    dtype=torch.float32).unsqueeze(0).to(device)

    pred_states = []
    with torch.no_grad():
        for k in range(horizon):
            next_target = torch.tensor(
                target_seq[H + k + 1], dtype=torch.float32).unsqueeze(0).to(device)
            next_state, _, _hx = predictor.predict(
                state, previous_state, target, previous_target, next_target, context)

            pred_states.append(next_state.squeeze(0).cpu().numpy())

            previous_state  = torch.cat([previous_state[:, 1:, :],  state.unsqueeze(1)], dim=1)
            previous_target = torch.cat([previous_target[:, 1:, :], target.unsqueeze(1)], dim=1)
            state  = next_state
            target = next_target

    return np.array(pred_states)  # (horizon, 6)


def rollout_with_controller(predictor, controller, state_seq, target_seq, context_seq, horizon):
    """使用 Controller 调整 target 后再滚动预测"""
    device = predictor.device
    H = horizon

    state               = torch.tensor(state_seq[H],              dtype=torch.float32).unsqueeze(0).to(device)
    previous_state      = torch.tensor(state_seq[:H],             dtype=torch.float32).unsqueeze(0).to(device)
    target              = torch.tensor(target_seq[H],             dtype=torch.float32).unsqueeze(0).to(device)
    previous_target     = torch.tensor(target_seq[:H],            dtype=torch.float32).unsqueeze(0).to(device)
    future_real_target  = torch.tensor(target_seq[H:H * 2],       dtype=torch.float32).unsqueeze(0).to(device)
    context             = torch.tensor(context_seq,               dtype=torch.float32).unsqueeze(0).to(device)

    # controller 一次性输出所有步的 delta
    with torch.no_grad():
        delta_targets = controller.control(
            state, previous_state, target, previous_target, future_real_target, context)
        # delta_targets: (1, controller_train_horizon, 3)

    pred_states = []
    ctrl_horizon = delta_targets.shape[1]

    with torch.no_grad():
        for k in range(horizon):
            next_real_target = torch.tensor(
                target_seq[H + k + 1], dtype=torch.float32).unsqueeze(0).to(device)

            # 在 controller 覆盖范围内使用调整后的 target，超出范围用原始 target
            if k < ctrl_horizon:
                next_target = next_real_target + delta_targets[:, k, :]
            else:
                next_target = next_real_target

            next_state, _, _hx = predictor.predict(
                state, previous_state, target, previous_target, next_target, context)

            pred_states.append(next_state.squeeze(0).cpu().numpy())

            previous_state  = torch.cat([previous_state[:, 1:, :],  state.unsqueeze(1)], dim=1)
            previous_target = torch.cat([previous_target[:, 1:, :], target.unsqueeze(1)], dim=1)
            state  = next_state
            target = next_target

    return np.array(pred_states)  # (horizon, 6)


def print_table(title, mae_per_step, mae_avg, train_horizon):
    dim_names = [d[3] for d in DENORM]
    header = f'{"step":>6}' + ''.join(f'{n:>12}' for n in dim_names)
    print(f'\n{title}')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for step in range(train_horizon):
        t = (step + 1) * 0.1
        vals = [f'{mae_per_step[step, dim_i] * (xmax - xmin) / 2.0:.4f}{unit}'
                for dim_i, (xmin, xmax, unit, _) in enumerate(DENORM)]
        print(f'{t:.1f}s'.rjust(6) + ''.join(f'{v:>12}' for v in vals))
    print('-' * len(header))
    avg_vals = [f'{mae_avg[dim_i] * (xmax - xmin) / 2.0:.4f}{unit}'
                for dim_i, (xmin, xmax, unit, _) in enumerate(DENORM)]
    print(f'{"avg":>6}' + ''.join(f'{v:>12}' for v in avg_vals))
    print('=' * len(header))


def validate(args):
    cfg = parse_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'))

    print(f'\n加载 Predictor: {args.predictor}')
    predictor = Predictor(cfg)
    predictor.load(args.predictor)
    predictor.eval()

    print(f'加载 Controller: {args.controller}')
    controller = Controller(cfg)
    controller.load(args.controller)
    controller.eval()

    print(f'\n读取验证数据: {args.data}')
    state_seqs, target_seqs, context_seqs = load_pt_data(args.data)
    N = len(state_seqs)
    print(f'共 {N} 个验证样本  state={tuple(state_seqs.shape)}  target={tuple(target_seqs.shape)}')

    H = cfg.horizon
    TH = cfg.train_horizon

    mae_base = np.zeros((N, TH, 3), dtype=np.float32)  # 只比较 v/a/kappa
    mae_ctrl = np.zeros((N, TH, 3), dtype=np.float32)

    print(f'\n运行验证（{N} 个样本）...')
    for i in range(N):
        # target: (27, 3) 取 [H+1:H+1+TH] → (TH, 3)
        target_vak = target_seqs[i, H + 1: H + 1 + TH, :].numpy()  # (TH, 3)

        pred_base = rollout_baseline(
            predictor, state_seqs[i], target_seqs[i], context_seqs[i], TH)
        pred_ctrl = rollout_with_controller(
            predictor, controller, state_seqs[i], target_seqs[i], context_seqs[i], TH)

        # 从 state 的 (TH, 6) 中提取 v/a/kappa (索引 3/4/5)
        pred_base_vak = pred_base[:, STATE_VAK_DIMS]  # (TH, 3)
        pred_ctrl_vak = pred_ctrl[:, STATE_VAK_DIMS]

        mae_base[i] = np.abs(pred_base_vak - target_vak)
        mae_ctrl[i] = np.abs(pred_ctrl_vak - target_vak)

    base_per_step = mae_base.mean(axis=0)   # (TH, 6)
    ctrl_per_step = mae_ctrl.mean(axis=0)
    base_avg      = mae_base.mean(axis=(0, 1))  # (6,)
    ctrl_avg      = mae_ctrl.mean(axis=(0, 1))

    print_table('Baseline（无 Controller）', base_per_step, base_avg, TH)
    print_table('With Controller',          ctrl_per_step, ctrl_avg, TH)

    # 提升幅度
    dim_names = [d[3] for d in DENORM]
    header = f'{"step":>6}' + ''.join(f'{n:>12}' for n in dim_names)
    print(f'\n提升幅度（正数=改善，负数=变差）')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for step in range(TH):
        t = (step + 1) * 0.1
        improve_vals = []
        for dim_i in range(3):
            b = base_per_step[step, dim_i]
            c = ctrl_per_step[step, dim_i]
            improve_vals.append(f'{(b - c) / (b + 1e-9) * 100:+.1f}%')
        print(f'{t:.1f}s'.rjust(6) + ''.join(f'{v:>12}' for v in improve_vals))
    print('-' * len(header))
    avg_improve = []
    for dim_i in range(3):
        b = base_avg[dim_i]
        c = ctrl_avg[dim_i]
        avg_improve.append(f'{(b - c) / (b + 1e-9) * 100:+.1f}%')
    print(f'{"avg":>6}' + ''.join(f'{v:>12}' for v in avg_improve))
    print('=' * len(header))

    if args.save:
        with open(args.save, 'w') as f:
            f.write('step,' +
                    ','.join(f'base_{d[3]}' for d in DENORM) + ',' +
                    ','.join(f'ctrl_{d[3]}' for d in DENORM) + '\n')
            for step in range(TH):
                t = (step + 1) * 0.1
                vals = [f'{t:.1f}']
                for dim_i, (xmin, xmax, _, _) in enumerate(DENORM):
                    vals.append(f'{base_per_step[step, dim_i] * (xmax - xmin) / 2.0:.6f}')
                for dim_i, (xmin, xmax, _, _) in enumerate(DENORM):
                    vals.append(f'{ctrl_per_step[step, dim_i] * (xmax - xmin) / 2.0:.6f}')
                f.write(','.join(vals) + '\n')
        print(f'\n结果已保存到 {args.save}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       '-f', required=True,            help='data_process.py 生成的 .pt 目录（含 state.pt/target.pt/context.pt）')
    parser.add_argument('--predictor',  '-p', required=True,            help='predictor 权重路径')
    parser.add_argument('--controller', '-con', required=True,            help='controller 权重路径')
    parser.add_argument('--save',       '-o', type=str, default=None,   help='将对比结果保存为 CSV')
    args = parser.parse_args()
    validate(args)
