"""
plot_trajectory.py
Plot desired and response trajectories from CSV data.

Usage:
  python3 tools/plot_trajectory.py --csv /path/to/data.csv
  python3 tools/plot_trajectory.py --csv /path/to/data.csv --frames 5 --step 200
  python3 tools/plot_trajectory.py --csv /path/to/data.csv --frame_idx 0 100 500
"""

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

DES_START       = 17
FIELDS_PER_DES  = 6   # x, y, s, v, a, kappa
FIELDS_PER_RESP = 5   # x, y, v, a, kappa
DES_COUNT       = 20
RESP_COUNT      = 20


def read_csv(path):
    rows = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                rows.append(row)
    return rows


def parse_trajectories(row):
    cur_x = float(row[1])
    cur_y = float(row[2])

    # desired trajectory
    des_x, des_y = [], []
    for i in range(DES_COUNT):
        base = DES_START + i * FIELDS_PER_DES
        des_x.append(float(row[base]))
        des_y.append(float(row[base + 1]))

    # response trajectory
    resp_start = DES_START + DES_COUNT * FIELDS_PER_DES
    resp_x, resp_y = [], []
    for i in range(RESP_COUNT):
        base = resp_start + i * FIELDS_PER_RESP
        resp_x.append(float(row[base]))
        resp_y.append(float(row[base + 1]))

    return cur_x, cur_y, des_x, des_y, resp_x, resp_y


def plot_frames(rows, frame_indices, save_path=None):
    n = len(frame_indices)
    cols = min(n, 3)
    rows_plot = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_plot, cols, figsize=(6 * cols, 5 * rows_plot))
    if n == 1:
        axes = [[axes]]
    elif rows_plot == 1:
        axes = [axes]

    for k, idx in enumerate(frame_indices):
        ax = axes[k // cols][k % cols]
        row = rows[idx]
        try:
            cur_x, cur_y, des_x, des_y, resp_x, resp_y = parse_trajectories(row)
        except Exception:
            ax.set_title(f'Frame {idx} (parse error)')
            continue

        ax.plot(des_x,  des_y,  'b-o', markersize=3, linewidth=1.5, label='Desired')
        ax.plot(resp_x, resp_y, 'r-o', markersize=3, linewidth=1.5, label='Response')
        ax.plot(cur_x,  cur_y,  'k*',  markersize=10,               label='Current')

        ax.set_title(f'Frame {idx}  (t={float(row[0]):.2f}s)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for k in range(n, rows_plot * cols):
        axes[k // cols][k % cols].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved to {save_path}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',       required=True,         help='Path to CSV file')
    parser.add_argument('--frames',    type=int, default=6,   help='Number of random frames to plot')
    parser.add_argument('--step',      type=int, default=None,help='Plot every N frames, e.g. --step 200')
    parser.add_argument('--frame_idx', type=int, nargs='+',   help='Specific frame indices, e.g. --frame_idx 0 100 500')
    parser.add_argument('--save',      type=str, default=None,help='Save figure to file, e.g. traj.png')
    args = parser.parse_args()

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
    plot_frames(rows, indices, save_path=args.save)


if __name__ == '__main__':
    main()
