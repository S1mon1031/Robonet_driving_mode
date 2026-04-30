"""
plot_va_interactive.py — 速度 / 加速度 交互式对比图（HTML）

绘制：
  - 实际车速 / 实际加速度（来自 CSV raw 字段）
  - 包内所有规划轨迹的 v / a（每条轨迹时间对齐到行时间戳）
  - 模型闭环预测的 v / a

输出可拖动缩放的 HTML 页面。

用法：
  cd /apollo/modules/MiningTruckTrack

  # 10s CSV（100点/行）
  python3 tools/plot_va_interactive.py \
      --csv /path/to/lap1_10s.csv \
      --model state_dict/10s/predictor_final.pth

  # 自定义配置
  python3 tools/plot_va_interactive.py \
      --csv /path/to/lap1_10s.csv \
      --model state_dict/10s/predictor_final.pth \
      --config config_10s.yaml \
      --csv_traj_points 100 \
      --max_steps 2000 \
      --stride 3 \
      --save logs/10s/va_interactive.html
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_model.parser import parse_config
from network_model.predictor import Predictor
from network_model.controller import Controller
from offline_train.data_process import (
    parse_row, normalize, compute_lateral_heading_error,
    LAT_MIN, LAT_MAX, S_MIN, S_MAX, HEAD_MIN, HEAD_MAX,
    V_MIN, V_MAX, A_MIN, A_MAX, KAPPA_MIN, KAPPA_MAX,
    PITCH_MIN, PITCH_MAX,
)


# ── 工具 ──────────────────────────────────────────────────────────────────────

def denorm(x, xmin, xmax):
    return (x + 1.0) * (xmax - xmin) / 2.0 + xmin


def state_norm(lat, s, head, v, a, kappa):
    def n(x, lo, hi): return normalize(np.clip(x, lo, hi), lo, hi)
    return [n(lat, LAT_MIN, LAT_MAX), n(s, S_MIN, S_MAX), n(head, HEAD_MIN, HEAD_MAX),
            n(v, V_MIN, V_MAX),       n(a, A_MIN, A_MAX), n(kappa, KAPPA_MIN, KAPPA_MAX)]


def target_norm(v_ref, a_ref, kappa_ref):
    def n(x, lo, hi): return normalize(np.clip(x, lo, hi), lo, hi)
    return [n(v_ref, V_MIN, V_MAX), n(a_ref, A_MIN, A_MAX), n(kappa_ref, KAPPA_MIN, KAPPA_MAX)]


# ── CSV 解析 ──────────────────────────────────────────────────────────────────

def parse_csv(path, csv_traj_points, max_steps=None):
    """解析 CSV，返回最长连续 D 档段 row 列表。"""
    segments, cur = [], []

    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for raw in reader:
            parsed = parse_row(raw, csv_traj_points)
            if parsed is None:
                if cur:
                    segments.append(cur)
                    cur = []
                continue
            gear = int(float(raw[10]))
            if gear == 0:
                if cur:
                    segments.append(cur)
                    cur = []
                continue
            try:
                cur_v = float(raw[5])
                cur_a = float(raw[6])
            except (IndexError, ValueError):
                if cur:
                    segments.append(cur)
                    cur = []
                continue

            ts, cur_x, cur_y, heading, pitch, load, desired, response = parsed
            cur.append({
                'ts':      ts,
                'ax':      cur_x,
                'ay':      cur_y,
                'heading': heading,
                'pitch':   pitch,
                'load':    load,
                'cur_v':   cur_v,
                'cur_a':   cur_a,
                'des':     desired,
                'des_x':   desired[0]['x'],
                'des_y':   desired[0]['y'],
                'des_v':   desired[0]['v'],
                'des_a':   desired[0]['a'],
                'des_k':   desired[0]['kappa'],
                'next_v':  desired[1]['v'] if len(desired) > 1 else desired[0]['v'],
                'next_a':  desired[1]['a'] if len(desired) > 1 else desired[0]['a'],
                'next_k':  desired[1]['kappa'] if len(desired) > 1 else desired[0]['kappa'],
                'kappa':   response[0]['kappa'],
            })

    if cur:
        segments.append(cur)
    if not segments:
        raise ValueError('未找到有效 D 档数据')

    longest = max(segments, key=len)
    print(f'CSV 共 {sum(len(s) for s in segments)} 行，最长连续段 {len(longest)} 行')
    if max_steps and len(longest) > max_steps:
        longest = longest[:max_steps]
        print(f'截断至前 {max_steps} 行')
    return longest


# ── 构建归一化数组（供闭环仿真使用） ─────────────────────────────────────────

def build_arrays(rows):
    N = len(rows)
    states   = np.zeros((N, 6), np.float32)
    targets  = np.zeros((N, 3), np.float32)
    contexts = np.zeros((N, 2), np.float32)

    for i, r in enumerate(rows):
        lat, s, head = compute_lateral_heading_error(
            r['ax'], r['ay'], r['heading'],
            r['des_x'], r['des_y'], r['heading'])
        states[i]   = state_norm(lat, s, head, r['cur_v'], r['cur_a'], r['kappa'])
        targets[i]  = target_norm(r['des_v'], r['des_a'], r['des_k'])
        contexts[i] = [normalize(np.clip(r['pitch'], PITCH_MIN, PITCH_MAX), PITCH_MIN, PITCH_MAX),
                       r['load']]

    return states, targets, contexts


# ── 闭环仿真 ──────────────────────────────────────────────────────────────────

def run_simulation(predictor, states, targets, contexts, H, closed_loop=False):
    """
    开环预测（默认）：每帧用真实历史状态预测下一步，误差不累积，适合评估模型单步精度。
    闭环预测：用预测值替换真实值滚动，适合评估长期稳定性（--closed_loop）。
    返回 pred_states (N-H-1, 6)。
    """
    device  = predictor.device
    N       = len(states)
    n_pred  = N - H - 1

    pred_states = np.zeros((n_pred, 6), np.float32)

    predictor.eval()
    with torch.no_grad():
        for step in range(n_pred):
            i = H + step

            if closed_loop and step > 0:
                prev_s = hist_s.copy()
                prev_t = hist_t.copy()
                cur_s  = pred_states[step - 1]
                cur_t  = targets[i]
            else:
                prev_s = states[i - H: i]
                prev_t = targets[i - H: i]
                cur_s  = states[i]
                cur_t  = targets[i]

            ps  = torch.tensor(prev_s,         dtype=torch.float32).unsqueeze(0).to(device)
            pt  = torch.tensor(prev_t,         dtype=torch.float32).unsqueeze(0).to(device)
            sn  = torch.tensor(cur_s,          dtype=torch.float32).unsqueeze(0).to(device)
            tn  = torch.tensor(cur_t,          dtype=torch.float32).unsqueeze(0).to(device)
            tnx = torch.tensor(targets[i + 1], dtype=torch.float32).unsqueeze(0).to(device)
            ctx = torch.tensor(contexts[i],    dtype=torch.float32).unsqueeze(0).to(device)

            next_s, _, _hx = predictor.predict(sn, ps, tn, pt, tnx, ctx)
            ns = next_s.squeeze(0).cpu().numpy()
            pred_states[step] = ns

            if closed_loop:
                if step == 0:
                    hist_s = states[i - H + 1: i + 1].copy()
                    hist_t = targets[i - H + 1: i + 1].copy()
                else:
                    hist_s = np.roll(hist_s, -1, axis=0); hist_s[-1] = cur_s
                    hist_t = np.roll(hist_t, -1, axis=0); hist_t[-1] = cur_t

            if (step + 1) % 500 == 0:
                print(f'  进度: {step+1}/{n_pred}')

    return pred_states


def _interp_history(arr, t_rows, t_end, H, dt=0.1):
    """
    以 dt 间隔从 arr（已按 t_rows 时间戳采样）中插值出 H 步历史，
    最后一个历史点对应 t_end - dt，第一个对应 t_end - H*dt。
    arr: (N, D)，t_rows: (N,)，均为相对时间
    """
    result = np.empty((H, arr.shape[1]), dtype=np.float32)
    for k in range(H):
        tq = t_end - (H - k) * dt
        if tq <= t_rows[0]:
            result[k] = arr[0]
        elif tq >= t_rows[-1]:
            result[k] = arr[-1]
        else:
            idx = int(np.searchsorted(t_rows, tq))
            idx = max(1, min(idx, len(t_rows) - 1))
            alpha = (tq - t_rows[idx - 1]) / (t_rows[idx] - t_rows[idx - 1] + 1e-9)
            result[k] = arr[idx - 1] + alpha * (arr[idx] - arr[idx - 1])
    return result


def run_rolling_horizon(predictor, states, targets, contexts, H,
                        rollout_steps, stride, t, rows, controller=None, delta_steps=5):
    """
    滚动短期闭环预测：每隔 stride 帧，从真实状态出发，闭环预测未来 rollout_steps 步。
    - prev_s / prev_t：以 0.1s 为步长从真实 CSV 行插值，保证与训练时历史长度一致（H×0.1s=3s）
    - next_target：同一行 desired 轨迹第 k+1 个点（0.1s 间隔）
    - x 轴：起始帧时间戳 + k×0.1s
    - 若传入 controller：额外绘制 with-controller 预测线
    """
    device = predictor.device
    N      = len(states)
    t_arr  = np.asarray(t, dtype=np.float32)  # 相对时间戳数组，行间距 ~0.2s

    x_v, y_v, x_a, y_a = [], [], [], []
    cx_v, cy_v, cx_a, cy_a = [], [], [], []
    dx_v, dy_dv = [], []   # delta_v (实际单位)
    dx_a, dy_da = [], []   # delta_a (实际单位)
    dx_s, dy_ds = [], []   # s_error with ctrl (实际单位)

    def des_target(row, k):
        """取同一行 desired 轨迹第 k 个点的归一化 target"""
        des = row['des']
        idx = min(k, len(des) - 1)
        d = des[idx]
        return target_norm(d['v'], d['a'], d['kappa'])

    predictor.eval()
    if controller is not None:
        controller.eval()

    n_seg = 0
    with torch.no_grad():
        for start in range(H, N - rollout_steps - 1, stride):
            t0    = float(t_arr[start])
            # 以 0.1s 步长插值出 H 步真实历史（覆盖 [t0-H*0.1, t0-0.1]）
            hist_s = _interp_history(states,  t_arr, t0, H, dt=0.1)
            hist_t = _interp_history(targets, t_arr, t0, H, dt=0.1)
            cur_s  = states[start].copy()
            cur_t  = np.array(des_target(rows[start], 0), dtype=np.float32)

            # with-controller 分支独立滚动状态
            hist_s_c = hist_s.copy()
            hist_t_c = hist_t.copy()
            cur_s_c  = cur_s.copy()
            cur_t_c  = cur_t.copy()

            for k in range(rollout_steps):
                # next_target 用同一行 desired 的第 k+1 个点（0.1s 间隔）
                tnx_val = np.array(des_target(rows[start], k + 1), dtype=np.float32)

                ps  = torch.tensor(hist_s,   dtype=torch.float32).unsqueeze(0).to(device)
                pt  = torch.tensor(hist_t,   dtype=torch.float32).unsqueeze(0).to(device)
                sn  = torch.tensor(cur_s,    dtype=torch.float32).unsqueeze(0).to(device)
                tn  = torch.tensor(cur_t,    dtype=torch.float32).unsqueeze(0).to(device)
                tnx = torch.tensor(tnx_val,  dtype=torch.float32).unsqueeze(0).to(device)
                ctx = torch.tensor(contexts[start], dtype=torch.float32).unsqueeze(0).to(device)

                next_s, _, _hx = predictor.predict(sn, ps, tn, pt, tnx, ctx)
                ns = next_s.squeeze(0).cpu().numpy()

                x_v.append(t0 + (k + 1) * 0.1)
                y_v.append(denorm(float(ns[3]), V_MIN, V_MAX))
                x_a.append(t0 + (k + 1) * 0.1)
                y_a.append(denorm(float(ns[4]), A_MIN, A_MAX))

                hist_s = np.roll(hist_s, -1, axis=0); hist_s[-1] = cur_s
                hist_t = np.roll(hist_t, -1, axis=0); hist_t[-1] = cur_t
                cur_s  = ns
                cur_t  = tnx_val

                # with-controller 预测（独立滚动）
                if controller is not None:
                    # 构造当前步的 future_real_target（长度 H）
                    fut = np.stack([
                        np.array(des_target(rows[start], k + 1 + h), dtype=np.float32)
                        for h in range(H)
                    ], axis=0)

                    ps_c  = torch.tensor(hist_s_c, dtype=torch.float32).unsqueeze(0).to(device)
                    pt_c  = torch.tensor(hist_t_c, dtype=torch.float32).unsqueeze(0).to(device)
                    sn_c  = torch.tensor(cur_s_c,  dtype=torch.float32).unsqueeze(0).to(device)
                    tn_c  = torch.tensor(cur_t_c,  dtype=torch.float32).unsqueeze(0).to(device)
                    fut_c = torch.tensor(fut,      dtype=torch.float32).unsqueeze(0).to(device)
                    ctx_c = torch.tensor(contexts[start], dtype=torch.float32).unsqueeze(0).to(device)

                    delta_targets = controller.control(sn_c, ps_c, tn_c, pt_c, fut_c, ctx_c)
                    # control() 一次性输出 controller_train_horizon 步，这里取第0步用于当前步
                    delta = delta_targets[:, 0, :].squeeze(0)
                    tnx_c = tn_c.squeeze(0) + delta
                    tnx_c = torch.clamp(tnx_c, -1.0, 1.0).unsqueeze(0)

                    next_s_c, _, _ = predictor.predict(sn_c, ps_c, tn_c, pt_c, tnx_c, ctx_c)
                    ns_c = next_s_c.squeeze(0).cpu().numpy()

                    cx_v.append(t0 + (k + 1) * 0.1)
                    cy_v.append(denorm(float(ns_c[3]), V_MIN, V_MAX))
                    cx_a.append(t0 + (k + 1) * 0.1)
                    cy_a.append(denorm(float(ns_c[4]), A_MIN, A_MAX))

                    # 记录前 delta_steps 步的 delta 和 s
                    if k < delta_steps:
                        tx = t0 + (k + 1) * 0.1
                        dv = denorm(float(delta[0]), V_MIN, V_MAX) - denorm(0.0, V_MIN, V_MAX)
                        da = denorm(float(delta[1]), A_MIN, A_MAX) - denorm(0.0, A_MIN, A_MAX)
                        ds = denorm(float(ns_c[1]), S_MIN, S_MAX)
                        dx_v.append(tx);  dy_dv.append(dv)
                        dx_a.append(tx);  dy_da.append(da)
                        dx_s.append(tx);  dy_ds.append(ds)

                    # with-controller 独立滚动
                    hist_s_c = np.roll(hist_s_c, -1, axis=0); hist_s_c[-1] = cur_s_c
                    hist_t_c = np.roll(hist_t_c, -1, axis=0); hist_t_c[-1] = cur_t_c
                    cur_s_c  = ns_c
                    cur_t_c  = tnx_c.squeeze(0).cpu().numpy()

            x_v.append(None); y_v.append(None)
            x_a.append(None); y_a.append(None)
            if controller is not None:
                cx_v.append(None); cy_v.append(None)
                cx_a.append(None); cy_a.append(None)
            n_seg += 1

    print(f'  滚动预测段数: {n_seg}')
    return x_v, y_v, x_a, y_a, cx_v, cy_v, cx_a, cy_a, dx_v, dy_dv, dx_a, dy_da, dx_s, dy_ds


# ── 构建规划轨迹 NaN-拼接数组（Plotly 单 trace 高效绘制）────────────────────

def build_plan_arrays(rows, stride, key):
    """
    把每隔 stride 行的规划轨迹，用 NaN 分隔拼成一维 x/y 数组，
    供 Plotly 单 Scatter trace 绘制。
    """
    ts = np.array([r['ts'] for r in rows])
    t  = ts - ts[0]

    x_all, y_all = [], []
    for i in range(0, len(rows), stride):
        r   = rows[i]
        t0  = t[i]
        des = r['des']
        for j, pt in enumerate(des):
            x_all.append(t0 + j * 0.1)
            y_all.append(pt[key])
        x_all.append(None)
        y_all.append(None)

    return x_all, y_all


# ── 绘制 HTML ─────────────────────────────────────────────────────────────────

def plot_html(rows, pred_states, H, save_path, stride, roll_x_v=None, roll_y_v=None,
              roll_x_a=None, roll_y_a=None,
              ctrl_x_v=None, ctrl_y_v=None, ctrl_x_a=None, ctrl_y_a=None,
              dx_v=None, dy_dv=None, dx_a=None, dy_da=None, dx_s=None, dy_ds=None):
    N  = len(rows)
    ts = np.array([r['ts'] for r in rows])
    t  = ts - ts[0]

    actual_v = np.array([r['cur_v'] for r in rows])
    actual_a = np.array([r['cur_a'] for r in rows])
    pitch    = np.array([r['pitch'] for r in rows])
    des_v0   = np.array([r['des'][0]['v'] for r in rows])
    des_a0   = np.array([r['des'][0]['a'] for r in rows])

    n_pred = len(pred_states)
    t_pred = t[H + 1: H + 1 + n_pred]
    pred_v = denorm(pred_states[:, 3], V_MIN, V_MAX)
    pred_a = denorm(pred_states[:, 4], A_MIN, A_MAX)

    pv_x, pv_y = build_plan_arrays(rows, stride, 'v')
    pa_x, pa_y = build_plan_arrays(rows, stride, 'a')
    n_plan = (N + stride - 1) // stride
    print(f'规划轨迹线条数: {n_plan}（stride={stride}）')

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.35, 0.35, 0.15, 0.15],
        subplot_titles=['Velocity (m/s)', 'Acceleration (m/s²)', 'Pitch (°)', 'Controller Delta'],
    )

    # ── 速度子图 ─────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=pv_x, y=pv_y,
        mode='lines',
        line=dict(color='rgba(255,152,0,0.18)', width=0.8),
        name='Planned v (all horizons)',
        legendgroup='plan_v',
        showlegend=True,
        hoverinfo='skip',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t.tolist(), y=des_v0.tolist(),
        mode='lines',
        line=dict(color='rgba(245,124,0,0.85)', width=1.5, dash='dash'),
        name='Planned v[0]',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_pred.tolist(), y=pred_v.tolist(),
        mode='lines',
        line=dict(color='rgba(244,67,54,0.90)', width=1.8),
        name='Predicted v (1-step)',
    ), row=1, col=1)

    if roll_x_v is not None:
        fig.add_trace(go.Scatter(
            x=roll_x_v, y=roll_y_v,
            mode='lines',
            line=dict(color='rgba(156,39,176,0.55)', width=1.2),
            name=f'Predicted v (3s rollout)',
            hoverinfo='skip',
        ), row=1, col=1)

    if ctrl_x_v is not None and len(ctrl_x_v) > 0:
        fig.add_trace(go.Scatter(
            x=ctrl_x_v, y=ctrl_y_v,
            mode='lines',
            line=dict(color='rgba(0,188,212,0.75)', width=1.2),
            name='Predicted v (with ctrl)',
            hoverinfo='skip',
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t.tolist(), y=actual_v.tolist(),
        mode='lines',
        line=dict(color='rgba(21,101,192,1.0)', width=2.0),
        name='Actual v',
    ), row=1, col=1)

    # ── 加速度子图 ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=pa_x, y=pa_y,
        mode='lines',
        line=dict(color='rgba(255,152,0,0.18)', width=0.8),
        name='Planned a (all horizons)',
        legendgroup='plan_a',
        showlegend=True,
        hoverinfo='skip',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t.tolist(), y=des_a0.tolist(),
        mode='lines',
        line=dict(color='rgba(229,115,0,0.85)', width=1.5, dash='dash'),
        name='Planned a[0]',
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t_pred.tolist(), y=pred_a.tolist(),
        mode='lines',
        line=dict(color='rgba(183,28,28,0.90)', width=1.8),
        name='Predicted a (1-step)',
    ), row=2, col=1)

    if roll_x_a is not None:
        fig.add_trace(go.Scatter(
            x=roll_x_a, y=roll_y_a,
            mode='lines',
            line=dict(color='rgba(103,58,183,0.55)', width=1.2),
            name='Predicted a (3s rollout)',
            hoverinfo='skip',
        ), row=2, col=1)

    if ctrl_x_a is not None and len(ctrl_x_a) > 0:
        fig.add_trace(go.Scatter(
            x=ctrl_x_a, y=ctrl_y_a,
            mode='lines',
            line=dict(color='rgba(0,150,136,0.75)', width=1.2),
            name='Predicted a (with ctrl)',
            hoverinfo='skip',
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t.tolist(), y=actual_a.tolist(),
        mode='lines',
        line=dict(color='rgba(198,40,40,1.0)', width=2.0),
        name='Actual a',
    ), row=2, col=1)

    fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'), row=2, col=1)

    # ── 坡度子图 ─────────────────────────────────────────────────────────────
    pitch_deg = np.degrees(pitch)
    fig.add_trace(go.Scatter(
        x=t.tolist(), y=pitch_deg.tolist(),
        mode='lines',
        line=dict(color='rgba(0,150,136,0.9)', width=1.5),
        name='Pitch (°)',
        fill='tozeroy',
        fillcolor='rgba(0,150,136,0.08)',
    ), row=3, col=1)

    fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'), row=3, col=1)

    # ── Controller Delta 子图 ────────────────────────────────────────────────
    if dx_v is not None and len(dx_v) > 0:
        fig.add_trace(go.Scatter(
            x=dx_v, y=dy_dv,
            mode='markers',
            marker=dict(color='rgba(0,188,212,0.6)', size=4),
            name='Δv (ctrl)',
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=dx_a, y=dy_da,
            mode='markers',
            marker=dict(color='rgba(0,150,136,0.6)', size=4),
            name='Δa (ctrl)',
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=dx_s, y=dy_ds,
            mode='markers',
            marker=dict(color='rgba(255,87,34,0.6)', size=4),
            name='s_err (with ctrl)',
        ), row=4, col=1)
        fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dot'), row=4, col=1)

    sim_start = float(t[H + 1])
    fig.add_vline(x=sim_start,
                  line=dict(color='green', width=1.5, dash='dot'),
                  annotation_text=f'Sim start ({sim_start:.1f}s)',
                  annotation_position='top right')

    fig.update_layout(
        title=dict(
            text='Velocity & Acceleration & Pitch — Actual / Planned / Predicted',
            font=dict(size=16),
            x=0.5,
        ),
        height=1100,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right',  x=1,
            font=dict(size=11),
        ),
        template='plotly_white',
        margin=dict(l=60, r=30, t=80, b=50),
    )
    fig.update_xaxes(title_text='Time (s)', row=4, col=1)
    fig.update_yaxes(title_text='Velocity (m/s)',      row=1, col=1)
    fig.update_yaxes(title_text='Acceleration (m/s²)', row=2, col=1)
    fig.update_yaxes(title_text='Pitch (°)',            row=3, col=1)
    fig.update_yaxes(title_text='Delta / s_err', row=4, col=1)

    fig.write_html(
        save_path,
        include_plotlyjs='cdn',
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'eraseshape'],
        },
    )
    print(f'\nHTML 已保存: {save_path}')


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',            type=str, required=True,
                        help='CSV 文件路径')
    parser.add_argument('--model',          type=str,
                        default=os.path.join(root, 'state_dict/10s/predictor_final.pth'),
                        help='predictor 权重路径（默认 state_dict/10s/predictor_final.pth）')
    parser.add_argument('--config', '-c',   type=str, default='config_10s.yaml',
                        help='配置文件（默认 config_10s.yaml）')
    parser.add_argument('--csv_traj_points', '-ctp', type=int, default=100,
                        help='CSV 每行轨迹点数，与采集时 --traj_duration 对应（默认 100 = 10s）')
    parser.add_argument('--max_steps',      type=int, default=3000,
                        help='最多使用行数（默认 3000 = 300s）')
    parser.add_argument('--stride',         type=int, default=1,
                        help='每隔多少行画一条规划轨迹（默认 1 = 全部）')
    parser.add_argument('--rollout',        type=int, default=0,
                        help='滚动短期闭环预测步数，0=不画（默认0）；'
                             '如 --rollout 30 每隔 --rollout_stride 帧画一段 3s 预测')
    parser.add_argument('--rollout_stride', type=int, default=10,
                        help='滚动预测的起始帧间隔（默认10=每1s画一条预测线）')
    parser.add_argument('--closed_loop',    action='store_true',
                        help='使用闭环仿真（默认开环：每帧用真实历史，误差不累积）')
    parser.add_argument('--controller',     type=str, default=None,
                        help='controller 权重路径，指定后在滚动预测中叠加 controller 修正（可选，需同时指定 --rollout）')
    parser.add_argument('--save',           type=str, default=None,
                        help='HTML 保存路径（默认 logs/10s/va_<csv名>.html）')
    args = parser.parse_args()

    cfg = parse_config(os.path.join(root, args.config))
    H   = cfg.horizon

    save_path = args.save or os.path.join(
        root, getattr(cfg, 'log_dir', 'logs/10s'),
        'va_' + os.path.splitext(os.path.basename(args.csv))[0] + '.html')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f'模型: {args.model}')
    predictor = Predictor(cfg)
    predictor.load(args.model)
    predictor.eval()

    controller = None
    if args.controller:
        print(f'Controller: {args.controller}')
        controller = Controller(cfg)
        controller.load(args.controller)
        controller.eval()

    print(f'CSV: {args.csv}  (csv_traj_points={args.csv_traj_points})')
    rows = parse_csv(args.csv, args.csv_traj_points, max_steps=args.max_steps)
    if len(rows) < H + 2:
        raise ValueError(f'有效行数 {len(rows)} 不足 horizon+2={H+2}')

    states, targets, contexts = build_arrays(rows)

    print(f'\n开始{"闭环" if args.closed_loop else "开环"}预测（共 {len(rows)-H-1} 步）...')
    pred_states = run_simulation(predictor, states, targets, contexts, H,
                                 closed_loop=args.closed_loop)

    roll_x_v = roll_y_v = roll_x_a = roll_y_a = None
    cx_v = cy_v = cx_a = cy_a = None
    dx_v = dy_dv = dx_a = dy_da = dx_s = dy_ds = None
    if args.rollout > 0:
        print(f'\n开始滚动 {args.rollout} 步闭环预测（每隔 {args.rollout_stride} 帧）...')
        roll_x_v, roll_y_v, roll_x_a, roll_y_a, cx_v, cy_v, cx_a, cy_a, dx_v, dy_dv, dx_a, dy_da, dx_s, dy_ds = run_rolling_horizon(
            predictor, states, targets, contexts, H,
            rollout_steps=args.rollout,
            stride=args.rollout_stride,
            t=np.array([r['ts'] for r in rows]) - rows[0]['ts'],
            rows=rows,
            controller=controller,
        )

    plot_html(rows, pred_states, H, save_path, stride=args.stride,
              roll_x_v=roll_x_v, roll_y_v=roll_y_v,
              roll_x_a=roll_x_a, roll_y_a=roll_y_a,
              ctrl_x_v=cx_v, ctrl_y_v=cy_v,
              ctrl_x_a=cx_a, ctrl_y_a=cy_a,
              dx_v=dx_v, dy_dv=dy_dv, dx_a=dx_a, dy_da=dy_da, dx_s=dx_s, dy_ds=dy_ds)


if __name__ == '__main__':
    main()
