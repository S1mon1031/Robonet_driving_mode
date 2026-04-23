"""
plot_prediction.py
Compare actual response trajectory vs predictor's predicted trajectory.

Usage:
  cd /apollo/modules/MiningTruckTrack
  python3 tools/plot_prediction.py --csv /path/to/data.csv --model state_dict/predictor_final.pth
  python3 tools/plot_prediction.py --csv /path/to/data.csv --model state_dict/predictor_final.pth --frames 6 --step 200
  python3 tools/plot_prediction.py --csv /path/to/data.csv --model state_dict/predictor_final.pth --frame_idx 0 100 500
"""

import csv
import os
import sys
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
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
    COL_DES_START as DES_START, FIELDS_PER_DES, FIELDS_PER_RESP,
)

DES_COUNT  = 20
RESP_COUNT = 20


def denorm(x, xmin, xmax):
    return (x + 1.0) * (xmax - xmin) / 2.0 + xmin


def frenet_to_xy(lat_err, s_err, ref_x, ref_y, ref_heading):
    """Frenet 误差 → 绝对坐标"""
    rx = ref_x + s_err * math.cos(ref_heading) - lat_err * math.sin(ref_heading)
    ry = ref_y + s_err * math.sin(ref_heading) + lat_err * math.cos(ref_heading)
    return rx, ry


def read_csv(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                rows.append(row)
    return rows


def parse_raw(row):
    """从原始 CSV 行提取期望轨迹和实际响应轨迹的 x, y"""
    cur_x = float(row[1])
    cur_y = float(row[2])
    heading = float(row[3])

    # desired trajectory: x, y, s, v, a, kappa
    des_xy, des_heading = [], []
    for i in range(DES_COUNT):
        base = DES_START + i * FIELDS_PER_DES
        dx, dy = float(row[base]), float(row[base + 1])
        des_xy.append((dx, dy))

    # compute heading from consecutive desired points
    for i in range(DES_COUNT):
        if i < DES_COUNT - 1:
            dx0, dy0 = des_xy[i]
            dx1, dy1 = des_xy[i + 1]
            dh = math.atan2(dy1 - dy0, dx1 - dx0)
        else:
            dh = des_heading[-1] if des_heading else heading
        des_heading.append(dh)

    # response trajectory: x, y, v, a, kappa
    resp_start = DES_START + DES_COUNT * FIELDS_PER_DES
    resp_xy = []
    for i in range(RESP_COUNT):
        base = resp_start + i * FIELDS_PER_RESP
        resp_xy.append((float(row[base]), float(row[base + 1])))

    return cur_x, cur_y, heading, des_xy, des_heading, resp_xy


def rollout(predictor, state_seq, target_seq, context_seq, des_xy, des_heading, horizon):
    """
    闭环预测，返回:
      pred_xy: [(rx, ry), ...]  长度 horizon
      pred_v:  [v, ...]         长度 horizon（真实单位 m/s）
      pred_a:  [a, ...]         长度 horizon（真实单位 m/s²）
    """
    device = predictor.device
    H = horizon

    state           = torch.tensor(state_seq[H],   dtype=torch.float32).unsqueeze(0).to(device)
    previous_state  = torch.tensor(state_seq[:H],  dtype=torch.float32).unsqueeze(0).to(device)
    target          = torch.tensor(target_seq[H],  dtype=torch.float32).unsqueeze(0).to(device)
    previous_target = torch.tensor(target_seq[:H], dtype=torch.float32).unsqueeze(0).to(device)
    context         = torch.tensor(context_seq,    dtype=torch.float32).unsqueeze(0).to(device)

    pred_xy, pred_v, pred_a = [], [], []
    with torch.no_grad():
        for k in range(H):
            next_target = torch.tensor(
                target_seq[H + k + 1], dtype=torch.float32).unsqueeze(0).to(device)
            next_state, _, _hx = predictor.predict(
                state, previous_state, target, previous_target, next_target, context)

            s = next_state.squeeze(0).cpu().numpy()
            lat_real = denorm(s[0], LAT_MIN, LAT_MAX)
            s_real   = denorm(s[1], S_MIN,   S_MAX)
            v_real   = denorm(s[3], V_MIN,   V_MAX)
            a_real   = denorm(s[4], A_MIN,   A_MAX)

            ref_idx = min(H + k + 1, len(des_xy) - 1)
            rx, ry = frenet_to_xy(lat_real, s_real,
                                  des_xy[ref_idx][0], des_xy[ref_idx][1],
                                  des_heading[ref_idx])
            pred_xy.append((rx, ry))
            pred_v.append(v_real)
            pred_a.append(a_real)

            previous_state  = torch.cat([previous_state[:, 1:, :],  state.unsqueeze(1)], dim=1)
            previous_target = torch.cat([previous_target[:, 1:, :], target.unsqueeze(1)], dim=1)
            state  = next_state
            target = next_target

    return pred_xy, pred_v, pred_a


def plot_frames(rows, frame_indices, predictor, cfg, save_path=None):
    H  = cfg.horizon
    TH = cfg.train_horizon
    steps = np.arange(1, TH + 1) * 0.1  # 时间轴 0.1s ~ 0.9s

    n = len(frame_indices)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]

    for k, idx in enumerate(frame_indices):
        ax_v, ax_a = axes[k]
        row = rows[idx]

        try:
            cur_x, cur_y, heading, des_xy, des_heading, resp_xy = parse_raw(row)
        except Exception:
            ax_xy.set_title(f'Frame {idx} (parse error)')
            continue

        result = process_row(row)
        if result is None:
            ax_xy.set_title(f'Frame {idx} (skipped)')
            continue
        s_arr, t_arr, c_arr = result

        j = TRAJ_POINTS // 2
        state_seq   = s_arr[j - H: j + H + 1]
        target_seq  = t_arr[max(0, j - H): j + 2 * H]
        context_seq = c_arr

        try:
            pred_xy, pred_v, pred_a = rollout(predictor, state_seq, target_seq, context_seq,
                                              des_xy, des_heading, TH)
        except Exception as e:
            ax_v.set_title(f'Frame {idx} (rollout error: {e})')
            continue

        # 实际响应的 v, a（从 state_seq 中心点往后取）
        real_v = [denorm(s_arr[j + 1 + i, 3], V_MIN, V_MAX) for i in range(TH)]
        real_a = [denorm(s_arr[j + 1 + i, 4], A_MIN, A_MAX) for i in range(TH)]
        # 期望 v, a（从 target_seq 中取）
        des_v  = [denorm(t_arr[j + 1 + i, 0], V_MIN, V_MAX) for i in range(TH)]
        des_a  = [denorm(t_arr[j + 1 + i, 1], A_MIN, A_MAX) for i in range(TH)]

        title = f'Frame {idx}  (t={float(row[0]):.2f}s)'

        # ── V 对比 ────────────────────────────────────────────
        ax_v.plot(steps, des_v,  'b--', linewidth=1.0, alpha=0.5, label='Desired v')
        ax_v.plot(steps, real_v, 'g-o', markersize=4,  linewidth=1.5, label='Actual v')
        ax_v.plot(steps, pred_v, 'r-o', markersize=4,  linewidth=1.5, label='Predicted v')
        ax_v.set_title(title)
        ax_v.set_xlabel('Time (s)')
        ax_v.set_ylabel('v (m/s)')
        ax_v.legend(fontsize=8)
        ax_v.grid(True, alpha=0.3)

        # ── A 对比 ────────────────────────────────────────────
        ax_a.plot(steps, des_a,  'b--', linewidth=1.0, alpha=0.5, label='Desired a')
        ax_a.plot(steps, real_a, 'g-o', markersize=4,  linewidth=1.5, label='Actual a')
        ax_a.plot(steps, pred_a, 'r-o', markersize=4,  linewidth=1.5, label='Predicted a')
        ax_a.set_title(title)
        ax_a.set_xlabel('Time (s)')
        ax_a.set_ylabel('a (m/s²)')
        ax_a.legend(fontsize=8)
        ax_a.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved to {save_path}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',       required=True,         help='Path to CSV file')
    parser.add_argument('--model',     required=True,         help='Predictor weight path')
    parser.add_argument('--frames',    type=int, default=6,   help='Number of random frames')
    parser.add_argument('--step',      type=int, default=None,help='Plot every N frames')
    parser.add_argument('--frame_idx', type=int, nargs='+',   help='Specific frame indices')
    parser.add_argument('--save',      type=str, default=None,help='Save figure to file')
    args = parser.parse_args()

    cfg = parse_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml'))
    predictor = Predictor(cfg)
    predictor.load(args.model)
    predictor.eval()
    print(f'Model loaded: {args.model}')

    print(f'Reading {args.csv} ...')
    rows = read_csv(args.csv)
    print(f'{len(rows)} rows total')

    if args.frame_idx:
        indices = [i for i in args.frame_idx if i < len(rows)]
    elif args.step:
        indices = list(range(0, len(rows), args.step))
    else:
        indices = sorted(np.random.choice(len(rows), min(args.frames, len(rows)), replace=False).tolist())

    print(f'Plotting frames: {indices}')
    plot_frames(rows, indices, predictor, cfg, save_path=args.save)


if __name__ == '__main__':
    main()
