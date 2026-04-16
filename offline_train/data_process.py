"""
data_process.py
从 collect_auto_driving_data.py 生成的 CSV 文件中提取训练数据，
转换为 PyTorch tensor 并保存到 data/ 目录。

使用方法：
  cd /apollo/modules/MiningTruckTrack
  python3 -m offline_train.data_process --csv <csv1> [csv2 ...] --outdir data/

示例（单个CSV）：
  python3 -m offline_train.data_process --csv /data/truck01_lap1.csv --outdir data/

示例（多个CSV）：
  python3 -m offline_train.data_process \
      --csv /data/truck01_lap1.csv /data/truck01_lap2.csv /data/truck02_lap1.csv \
      --outdir data/

输出文件：
  data/state.pt       (N, 2*H+1, 6)  — (lateral_error, s_error, heading_error, v, a, kappa)
  data/target.pt      (N, 3*H,   3)  — (v_ref, a_ref, kappa_ref)
  data/real_target.pt (N, 3*H,   3)  — 同 target
  data/context.pt     (N, 2)         — (pitch, load)

  你的 10s CSV（100点/行，直接模式）：                                                                                                                                                                             
  cd /apollo/modules/MiningTruckTrack                                                                                                                                                                              
  python3 -m offline_train.data_process \                                                                                                                                                                          
      --csv /path/to/your_10s.csv \                                                                                                                                                                                
      --outdir data/ \                                                                                                                                                                                           
      --csv_traj_points 100 \
      --traj_points 100

  旧 2s CSV 拼成 5s（拼接模式，向下兼容）：
  python3 -m offline_train.data_process \
      --csv /path/to/old_2s.csv \
      --outdir data/ \
      --csv_traj_points 20 \
      --traj_points 50

"""

import os
import sys
import math
import csv
import argparse
import numpy as np
import torch


# ── 数值裁剪与归一化范围 ─────────────────────────────────────────────────────
# state: (lateral_error, s_error, heading_error, v, a, kappa)
LAT_MIN,   LAT_MAX   = -1.0,   1.0
S_MIN,     S_MAX     = -5.0,   5.0
HEAD_MIN,  HEAD_MAX  = -math.radians(10), math.radians(10)  # ±0.1745 rad
V_MIN,     V_MAX     = -3.0,  11.0
A_MIN,     A_MAX     = -3.0,   3.0
KAPPA_MIN, KAPPA_MAX = -0.1,   0.1
# context: (pitch, load)
PITCH_MIN, PITCH_MAX = -math.radians(8), math.radians(8)    # ±0.1396 rad

# 归一化到 [-1, 1]: norm = 2 * (x - min) / (max - min) - 1
STATE_MINS  = None  # 运行时构建
TARGET_MINS = None


def normalize(x, xmin, xmax):
    return 2.0 * (x - xmin) / (xmax - xmin + 1e-8) - 1.0


HORIZON     = 9     # 历史/未来窗口步数
TRAJ_POINTS = 20    # 每条轨迹的时间步数（0.1s × 20 = 2s）
STATE_DIM   = 6     # lateral_error, s_error, heading_error, v, a, kappa
TARGET_DIM  = 3     # v_ref, a_ref, kappa_ref
CONTEXT_DIM = 2     # pitch, load

# CSV 列索引（与 collect_auto_driving_data.py 的表头对应）
COL_TIMESTAMP   = 0
COL_CURRENT_X   = 1
COL_CURRENT_Y   = 2
COL_HEADING     = 3
COL_PITCH       = 4
COL_GEAR        = 10  # gear_location: 0=N, -1=R, 1~6=D档
# current_v=5, current_a=6, chassis: 7~13 (engine_torque在13), control: 14~16
COL_LOAD        = 16   # control_backup_field_a 解析后的 load
# desired_trajectory 从 17 开始，每点 6 个值: x,y,s,v,a,kappa
COL_DES_START   = 17
COL_DES_COUNT   = COL_DES_START  # 最后一个 desired_traj_count 位置动态计算
# response_trajectory 紧随其后
FIELDS_PER_DES  = 6   # x,y,s,v,a,kappa
FIELDS_PER_RESP = 5   # x,y,v,a,kappa


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def compute_lateral_heading_error(rx, ry, rh, dx, dy, dh):
    """
    计算实际位置相对于期望位置的横向误差、纵向误差和航向误差
    rx,ry,rh: 实际 x,y,heading
    dx,dy,dh: 期望 x,y,heading
    lateral_error: 正值表示偏左（相对期望航向法线方向）
    s_error:       正值表示超前（相对期望航向切线方向）
    """
    ex = rx - dx
    ey = ry - dy
    lateral_error = -ex * math.sin(dh) + ey * math.cos(dh)
    s_error        =  ex * math.cos(dh) + ey * math.sin(dh)
    heading_error  = normalize_angle(rh - dh)
    return lateral_error, s_error, heading_error


def resample_trajectory(points_t, points_val, target_times):
    """
    线性插值将变时间步轨迹重采样到均匀时间步
    points_t:   原始时间列表
    points_val: 原始值列表（与 points_t 等长）
    target_times: 目标时间列表
    """
    result = []
    for t in target_times:
        if t <= points_t[0]:
            result.append(points_val[0])
            continue
        if t >= points_t[-1]:
            result.append(points_val[-1])
            continue
        # 二分查找
        lo, hi = 0, len(points_t) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if points_t[mid] <= t:
                lo = mid
            else:
                hi = mid
        ratio = (t - points_t[lo]) / (points_t[hi] - points_t[lo] + 1e-9)
        interp = [points_val[lo][j] + ratio * (points_val[hi][j] - points_val[lo][j])
                  for j in range(len(points_val[0]))]
        result.append(interp)
    return result


def parse_row(row, csv_traj_points=20):
    """解析 CSV 一行，返回所需字段，失败返回 None"""
    try:
        timestamp  = float(row[COL_TIMESTAMP])
        current_x  = float(row[COL_CURRENT_X])
        current_y  = float(row[COL_CURRENT_Y])
        heading    = float(row[COL_HEADING])
        pitch      = float(row[COL_PITCH])
        load       = float(row[COL_LOAD])

        des_count = csv_traj_points
        des_fields = des_count * FIELDS_PER_DES
        resp_count = csv_traj_points
        resp_fields = resp_count * FIELDS_PER_RESP

        # 期望轨迹
        des_raw = row[COL_DES_START: COL_DES_START + des_fields]
        desired = []
        for i in range(des_count):
            base = i * FIELDS_PER_DES
            desired.append({
                'x':     float(des_raw[base]),
                'y':     float(des_raw[base + 1]),
                's':     float(des_raw[base + 2]),
                'v':     float(des_raw[base + 3]),
                'a':     float(des_raw[base + 4]),
                'kappa': float(des_raw[base + 5]),
            })

        # 响应轨迹
        resp_start = COL_DES_START + des_fields
        resp_raw = row[resp_start: resp_start + resp_fields]
        response = []
        for i in range(resp_count):
            base = i * FIELDS_PER_RESP
            response.append({
                'x':     float(resp_raw[base]),
                'y':     float(resp_raw[base + 1]),
                'v':     float(resp_raw[base + 2]),
                'a':     float(resp_raw[base + 3]),
                'kappa': float(resp_raw[base + 4]),
                't':     round((i + 1) * 0.1, 1),  # t直接推算，0.1s间隔
            })

        return timestamp, current_x, current_y, heading, pitch, load, desired, response
    except Exception:
        return None


def process_row(row, csv_traj_points=20):
    """
    将一行 CSV 转换为均匀时间步的 state 序列和 target 序列
    返回:
      states:  (csv_traj_points, STATE_DIM)
      targets: (csv_traj_points, TARGET_DIM)
      context: (CONTEXT_DIM,)
    失败返回 None
    """
    parsed = parse_row(row, csv_traj_points)
    if parsed is None:
        return None
    timestamp, cur_x, cur_y, heading, pitch, load, desired, response = parsed

    # N档数据跳过
    if int(float(row[COL_GEAR])) == 0:
        return None

    if len(desired) < 2 or len(response) < 2:
        return None

    # 均匀目标时间步 [0.1, 0.2, ..., N*0.1]
    target_times = [round((i + 1) * 0.1, 1) for i in range(csv_traj_points)]

    # ── 期望轨迹插值 ──────────────────────────────────────────────
    # 第一个期望点的 relative_time 已对齐到0，用 s 或 v 对应时间需要我们手动推
    # 期望轨迹用其自身的索引推算时间（planning 100ms一个点）
    des_times = [round((i) * 0.1, 1) for i in range(len(desired))]  # 0, 0.1, ..., 2.0
    des_vals  = [[p['v'], p['a'], p['kappa']] for p in desired]
    des_pos   = [[p['x'], p['y'], p['x']] for p in desired]  # 用于计算误差

    # 插值
    des_interp     = resample_trajectory(des_times, des_vals,    target_times)
    des_pos_interp = resample_trajectory(des_times,
                                         [[p['x'], p['y'], p.get('heading', heading)]
                                          for p in desired],
                                         target_times)

    # ── 响应轨迹插值 ──────────────────────────────────────────────
    resp_times = [p['t'] for p in response]
    resp_vals  = [[p['x'], p['y'], p['v'], p['a'], p['kappa']] for p in response]
    resp_interp = resample_trajectory(resp_times, resp_vals, target_times)

    # ── 构造 state/target 序列 ─────────────────────────────────────
    states  = []
    targets = []

    for i in range(csv_traj_points):
        rx, ry, rv, ra, rk = resp_interp[i]
        dx, dy, dh_approx  = des_pos_interp[i]
        dv, da, dk         = des_interp[i]

        # heading：用期望点的航向近似（期望轨迹没有单独存 heading，用当前帧 heading 近似）
        lat_err, s_err, head_err = compute_lateral_heading_error(rx, ry, heading, dx, dy, heading)

        # clip
        lat_err  = float(np.clip(lat_err,  LAT_MIN,   LAT_MAX))
        s_err    = float(np.clip(s_err,    S_MIN,     S_MAX))
        head_err = float(np.clip(head_err, HEAD_MIN,  HEAD_MAX))
        rv = float(np.clip(rv, V_MIN,     V_MAX))
        ra = float(np.clip(ra, A_MIN,     A_MAX))
        rk = float(np.clip(rk, KAPPA_MIN, KAPPA_MAX))
        dv = float(np.clip(dv, V_MIN,     V_MAX))
        da = float(np.clip(da, A_MIN,     A_MAX))
        dk = float(np.clip(dk, KAPPA_MIN, KAPPA_MAX))

        # 归一化到 [-1, 1]
        lat_err  = normalize(lat_err,  LAT_MIN,   LAT_MAX)
        s_err    = normalize(s_err,    S_MIN,     S_MAX)
        head_err = normalize(head_err, HEAD_MIN,  HEAD_MAX)
        rv = normalize(rv, V_MIN,     V_MAX)
        ra = normalize(ra, A_MIN,     A_MAX)
        rk = normalize(rk, KAPPA_MIN, KAPPA_MAX)
        dv = normalize(dv, V_MIN,     V_MAX)
        da = normalize(da, A_MIN,     A_MAX)
        dk = normalize(dk, KAPPA_MIN, KAPPA_MAX)

        states.append([lat_err, s_err, head_err, rv, ra, rk])
        targets.append([dv, da, dk])

    pitch_norm = normalize(float(np.clip(pitch, PITCH_MIN, PITCH_MAX)), PITCH_MIN, PITCH_MAX)
    context = [pitch_norm, load]  # load(0/1)保持不变
    return (np.array(states,  dtype=np.float32),   # (TRAJ_POINTS, STATE_DIM)
            np.array(targets, dtype=np.float32),   # (TRAJ_POINTS, TARGET_DIM)
            np.array(context, dtype=np.float32))   # (CONTEXT_DIM,)


def build_sequences(all_states, all_targets, all_contexts, horizon, traj_points=None):
    """
    将每条轨迹的时间步序列，构建滑动窗口训练样本

    对于第 i 条轨迹的第 j 步（horizon <= j < traj_points - horizon）：
      state_seq:   steps [j-H, ..., j, ..., j+H]  (2H+1, state_dim)
      target_seq:  steps [j-H, ..., j, ..., j+2H-1] (3H, target_dim)
      context:     当前行的 context                  (context_dim,)
    """
    if traj_points is None:
        traj_points = TRAJ_POINTS
    H = horizon
    state_seqs, target_seqs, real_target_seqs, contexts = [], [], [], []

    for i in range(len(all_states)):
        S = all_states[i]    # (traj_points, state_dim)
        T = all_targets[i]   # (traj_points, target_dim)
        C = all_contexts[i]  # (context_dim,)

        for j in range(H, traj_points - H):
            state_seq  = S[j - H: j + H + 1]          # (2H+1, state_dim)
            target_seq = T[max(0, j - H): j + 2 * H]  # (3H, target_dim)

            # 边界补齐
            if j < H:
                pad = np.tile(T[0], (H - j, 1))
                target_seq = np.concatenate([pad, target_seq], axis=0)
            if j + 2 * H > traj_points:
                pad = np.tile(T[-1], (j + 2 * H - traj_points, 1))
                target_seq = np.concatenate([target_seq, pad], axis=0)

            state_seqs.append(state_seq)
            target_seqs.append(target_seq)
            real_target_seqs.append(target_seq.copy())
            contexts.append(C)

    return (np.array(state_seqs,       dtype=np.float32),
            np.array(target_seqs,      dtype=np.float32),
            np.array(real_target_seqs, dtype=np.float32),
            np.array(contexts,         dtype=np.float32))


def load_and_stitch(csv_path, target_traj_points, csv_traj_points=20):
    """
    读取 CSV 并生成目标长度的轨迹序列。

    两种模式：
    1. 直接模式（csv_traj_points == target_traj_points）：
       CSV 每行已包含足够长的轨迹，直接解析每行即可。
    2. 拼接模式（csv_traj_points < target_traj_points）：
       CSV 每行是短轨迹，拼接相邻行生成更长序列。
       拼接规则（以 csv=20, target=50 为例）：
         当前行：  点 0-19  → t=0.1s~2.0s
         T+1s 行： 点 10-19 → t=2.1s~3.0s
         ...
    """
    BASE  = csv_traj_points
    STEP  = BASE // 2
    extra = (target_traj_points - BASE) // STEP
    ts_gap = STEP * 0.1

    # 读取并按时间戳排序
    raw = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                try:
                    raw.append((float(row[COL_TIMESTAMP]), row))
                except Exception:
                    pass
    raw.sort(key=lambda x: x[0])
    n = len(raw)

    all_states, all_targets, all_contexts = [], [], []
    skip = 0

    for i in range(n):
        ts_i, row_i = raw[i]

        result_i = process_row(row_i, csv_traj_points)
        if result_i is None:
            skip += 1
            continue
        states_i, targets_i, ctx_i = result_i

        if extra == 0:  # 直接模式：CSV 已有足够长轨迹
            all_states.append(states_i)
            all_targets.append(targets_i)
            all_contexts.append(ctx_i)
            continue

        seg_s = [states_i]
        seg_t = [targets_i]
        ok = True

        for r in range(1, extra + 1):
            target_ts = ts_i + r * ts_gap
            # 在 i 附近搜索最近时间戳的行
            best_j, best_diff = None, float('inf')
            lo = max(0, i + r * STEP - 3)
            hi = min(n, i + r * STEP + 4)
            for j in range(lo, hi):
                d = abs(raw[j][0] - target_ts)
                if d < best_diff:
                    best_diff, best_j = d, j

            # 时间容差 300ms，且中间不能有超过 500ms 的跳变（跨 session）
            if best_j is None or best_diff > 0.3:
                ok = False
                break
            for k in range(i, best_j):
                if raw[k + 1][0] - raw[k][0] > 0.5:
                    ok = False
                    break
            if not ok:
                break

            res_j = process_row(raw[best_j][1], csv_traj_points)
            if res_j is None:
                ok = False
                break
            states_j, targets_j, _ = res_j
            seg_s.append(states_j[STEP:])   # 取后半段
            seg_t.append(targets_j[STEP:])

        if not ok:
            skip += 1
            continue

        all_states.append(np.concatenate(seg_s, axis=0))   # (target_traj_points, 6)
        all_targets.append(np.concatenate(seg_t, axis=0))  # (target_traj_points, 3)
        all_contexts.append(ctx_i)

    return all_states, all_targets, all_contexts, skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', '-f', required=True, nargs='+', help='输入 CSV 文件路径（支持多个）')
    parser.add_argument('--outdir', '-o', default='./data', help='输出目录')
    parser.add_argument('--traj_points', '-tp', type=int, default=50,
                        help='目标轨迹点数，如 50=5s, 100=10s（默认50）')
    parser.add_argument('--csv_traj_points', '-ctp', type=int, default=20,
                        help='CSV 每行的轨迹点数，与 collect 脚本的 --traj_duration 对应，'
                             '如 --traj_duration 2.0→20, 5.0→50, 10.0→100（默认20）')
    parser.add_argument('--horizon', '-hz', type=int, default=HORIZON,
                        help=f'滑动窗口半径，需与 config.yaml 的 horizon 一致（默认{HORIZON}）')
    args = parser.parse_args()

    traj_points     = args.traj_points
    csv_traj_points = args.csv_traj_points
    horizon         = args.horizon

    if traj_points < csv_traj_points:
        raise ValueError(f'--traj_points({traj_points}) 不能小于 --csv_traj_points({csv_traj_points})')
    if traj_points > csv_traj_points:
        # 拼接模式：需要能整除
        step = csv_traj_points // 2
        if (traj_points - csv_traj_points) % step != 0:
            raise ValueError(
                f'拼接模式下 (traj_points - csv_traj_points) 需能被 {step} 整除，'
                f'当前: ({traj_points} - {csv_traj_points}) = {traj_points - csv_traj_points}')

    os.makedirs(args.outdir, exist_ok=True)

    all_states, all_targets, all_contexts = [], [], []
    total_skip = 0

    for csv_path in args.csv:
        print(f'读取 {csv_path} ...')
        s, t, c, skip = load_and_stitch(csv_path, traj_points, csv_traj_points)
        print(f'  有效轨迹 {len(s)} 条，跳过 {skip} 条')
        all_states.extend(s)
        all_targets.extend(t)
        all_contexts.extend(c)
        total_skip += skip

    print(f'合计 {len(all_states)} 条有效轨迹，跳过 {total_skip} 条')

    state_arr, target_arr, real_target_arr, context_arr = build_sequences(
        all_states, all_targets, all_contexts, horizon, traj_points)

    print(f'生成训练样本: {state_arr.shape[0]}')
    print(f'  state_seq:       {state_arr.shape}')
    print(f'  target_seq:      {target_arr.shape}')
    print(f'  real_target_seq: {real_target_arr.shape}')
    print(f'  context:         {context_arr.shape}')

    torch.save(torch.from_numpy(state_arr),       os.path.join(args.outdir, 'state.pt'))
    torch.save(torch.from_numpy(target_arr),      os.path.join(args.outdir, 'target.pt'))
    torch.save(torch.from_numpy(real_target_arr), os.path.join(args.outdir, 'real_target.pt'))
    torch.save(torch.from_numpy(context_arr),     os.path.join(args.outdir, 'context.pt'))

    # 保存归一化参数（推理时使用）
    norm_stats = {
        'lat_min': LAT_MIN,   'lat_max': LAT_MAX,
        's_min':   S_MIN,     's_max':   S_MAX,
        'head_min': HEAD_MIN, 'head_max': HEAD_MAX,
        'v_min':   V_MIN,     'v_max':   V_MAX,
        'a_min':   A_MIN,     'a_max':   A_MAX,
        'kappa_min': KAPPA_MIN, 'kappa_max': KAPPA_MAX,
        'pitch_min': PITCH_MIN, 'pitch_max': PITCH_MAX,
    }
    torch.save(norm_stats, os.path.join(args.outdir, 'norm_stats.pt'))
    print(f'已保存至 {args.outdir}/')


if __name__ == '__main__':
    main()
