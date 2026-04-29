"""
plot_train_log.py
解析训练日志文件并绘制 loss 曲线。

支持同时传入多个日志文件（自动拼接续训 epoch），也支持单文件。

使用方法：
  cd /apollo/modules/MiningTruckTrack
  python3 tools/plot_train_log.py --log logs/10s_h30_lstm/train_log_*.txt
  python3 tools/plot_train_log.py --log logs/10s_h30_lstm/train_log_20260423_172015.txt
  python3 tools/plot_train_log.py --log log1.txt log2.txt --out loss.png
"""

import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── 正则表达式 ──────────────────────────────────────────────────────────────
# Predictor: [17:48:25] Epoch    1 (24.7s) | loss=0.0052 | grad=2.465 | lat=0.2168 s=0.0423 v=0.0653 a=0.0091
RE_PRED = re.compile(
    r'Epoch\s+(\d+).*?loss=([\d.]+).*?grad=([\d.]+).*?'
    r'lat=([\d.]+)\s+s=([\d.]+)\s+v=([\d.]+)\s+a=([\d.]+)'
)
# Controller: [xx:xx:xx] Epoch    1 (xs) | track=0.0012 smooth=0.0003 total=0.0015 | dv=0.1234 da=0.0234 dkappa=0.0123
RE_CTRL = re.compile(
    r'Epoch\s+(\d+).*?track=([\d.]+)\s+smooth=([\d.]+)\s+total=([\d.]+)'
)


def parse_logs(log_paths):
    pred_rows = []   # (epoch_global, loss, grad, lat, s, v, a)
    ctrl_rows = []   # (epoch_global, track, smooth, total, grad)

    pred_offset = 0
    ctrl_offset = 0
    in_pred = False
    in_ctrl = False

    for path in sorted(log_paths):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        file_pred = []
        file_ctrl = []

        for line in lines:
            if 'Predictor' in line and 'epochs' in line:
                in_pred = True
                in_ctrl = False
                continue
            if 'Controller' in line and 'epochs' in line:
                in_pred = False
                in_ctrl = True
                continue

            m = RE_PRED.search(line)
            if m and in_pred:
                ep = int(m.group(1))
                file_pred.append((ep, float(m.group(2)), float(m.group(3)),
                                  float(m.group(4)), float(m.group(5)),
                                  float(m.group(6)), float(m.group(7))))
                continue

            m = RE_CTRL.search(line)
            if m and in_ctrl:
                ep = int(m.group(1))
                file_ctrl.append((ep, float(m.group(2)), float(m.group(3)),
                                  float(m.group(4))))

        # 拼接：用 offset 保证跨文件 epoch 连续
        for row in file_pred:
            pred_rows.append((row[0] + pred_offset,) + row[1:])
        for row in file_ctrl:
            ctrl_rows.append((row[0] + ctrl_offset,) + row[1:])

        if file_pred:
            pred_offset = pred_rows[-1][0]
        if file_ctrl:
            ctrl_offset = ctrl_rows[-1][0]

    return pred_rows, ctrl_rows


def plot(pred_rows, ctrl_rows, out_path):
    has_pred = len(pred_rows) > 0
    has_ctrl = len(ctrl_rows) > 0
    n_plots = has_pred + has_ctrl
    if n_plots == 0:
        print('未找到任何训练记录，请检查日志格式。')
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots), squeeze=False)
    ax_idx = 0

    # ── Predictor ────────────────────────────────────────────────────────────
    if has_pred:
        epochs = [r[0] for r in pred_rows]
        loss   = [r[1] for r in pred_rows]
        grad   = [r[2] for r in pred_rows]
        lat    = [r[3] for r in pred_rows]
        s      = [r[4] for r in pred_rows]
        v      = [r[5] for r in pred_rows]
        a      = [r[6] for r in pred_rows]

        ax = axes[ax_idx][0]
        ax2 = ax.twinx()

        ax.plot(epochs, loss, 'b-o', markersize=3, label='loss')
        ax2.plot(epochs, s,    'g--^', markersize=3, alpha=0.7, label='s MAE')
        ax2.plot(epochs, v,    'm--D', markersize=3, alpha=0.7, label='v MAE')
        ax2.plot(epochs, a,    'c--x', markersize=3, alpha=0.7, label='a MAE')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color='b')
        ax2.set_ylabel('MAE (归一化)', color='r')
        ax.set_title('Predictor Training')
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        ax_idx += 1

    # ── Controller ───────────────────────────────────────────────────────────
    if has_ctrl:
        epochs = [r[0] for r in ctrl_rows]
        track  = [r[1] for r in ctrl_rows]
        smooth = [r[2] for r in ctrl_rows]
        total  = [r[3] for r in ctrl_rows]

        ax = axes[ax_idx][0]
        ax.plot(epochs, total,  'b-o',  markersize=3, label='total loss')
        ax.plot(epochs, track,  'r--s', markersize=3, alpha=0.8, label='track loss')
        ax.plot(epochs, smooth, 'g--^', markersize=3, alpha=0.8, label='smooth loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Controller Training')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        ax_idx += 1

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f'已保存: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-l', nargs='+', required=True,
                        help='日志文件路径，支持多个（按文件名排序后拼接）')
    parser.add_argument('--out', '-o', default='train_loss.png',
                        help='输出图片路径（默认 train_loss.png）')
    args = parser.parse_args()

    pred_rows, ctrl_rows = parse_logs(args.log)
    print(f'Predictor: {len(pred_rows)} epochs，Controller: {len(ctrl_rows)} epochs')
    plot(pred_rows, ctrl_rows, args.out)


if __name__ == '__main__':
    main()
