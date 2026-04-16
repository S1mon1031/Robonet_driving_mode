import os
import sys
import argparse
import time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network_model.parser import parse_config
from network_model.predictor import Predictor
from network_model.controller import Controller
from offline_train.container import Container


def log(msg, log_file):
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='配置文件路径，相对于项目根目录，如 config_10s.yaml（默认 config.yaml）')
    parser.add_argument('--data_dir', '-d', type=str, nargs='+', default=['data'],
                        help='训练数据目录，支持多个，如 data/10s/part1 data/10s/part2（默认 data）')
    parser.add_argument('--resume_predictor', '-lpre', type=str, default=None,
                        help='从已有 predictor 权重继续训练，如 state_dict/predictor_500.pth')
    parser.add_argument('--resume_controller', '-lcon', type=str, default=None,
                        help='从已有 controller 权重继续训练')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(__file__))
    cfg = parse_config(os.path.join(root, args.config))

    data_dirs = [os.path.join(root, d) for d in args.data_dir]
    for d in data_dirs:
        for fname in ['state.pt', 'target.pt', 'real_target.pt', 'context.pt']:
            fp = os.path.join(d, fname)
            if not os.path.exists(fp):
                raise FileNotFoundError(
                    f'{fp} 不存在，请先运行:\n'
                    f'  python3 -m offline_train.data_process --csv <your.csv> --outdir {d}')

    container = Container(data_dirs, cfg.batch_size)
    predictor  = Predictor(cfg)
    controller = Controller(cfg)

    # 优先读 config 中的路径，否则自动从 data_dir 推断
    save_dir = os.path.join(root, getattr(cfg, 'save_dir', None) or
                            os.path.join('state_dict', os.path.basename(data_dirs[0].rstrip('/'))))
    log_dir  = os.path.join(root, getattr(cfg, 'log_dir',  None) or
                            os.path.join('logs',       os.path.basename(data_dirs[0].rstrip('/'))))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)
    log_file = os.path.join(log_dir, f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    if args.resume_predictor:
        predictor.load(args.resume_predictor)
        log(f'已加载 predictor 权重: {args.resume_predictor}', log_file)

    if args.resume_controller:
        controller.load(args.resume_controller)
        log(f'已加载 controller 权重: {args.resume_controller}', log_file)

    # ── 第一阶段：训练 predictor ────────────────────────────────────────────
    log(f'\n=== 训练 Predictor（{cfg.predictor_epochs} epochs）===', log_file)
    log(f'参数量: {predictor.total_params}', log_file)

    for e in range(1, cfg.predictor_epochs + 1):
        t0 = time.time()
        epoch_end = False
        while not epoch_end:
            epoch_end, metrics, tsm = predictor.update(container)
        elapsed = time.time() - t0

        now = datetime.now().strftime('%H:%M:%S')
        msg = (f'[{now}] Epoch {e:4d} ({elapsed:.1f}s) | loss={metrics["total_loss"]:.4f} '
               f'| grad={metrics["grad_norm"]:.3f}'
               + (f' | lat={tsm["lat"][-1]:.4f} s={tsm["s"][-1]:.4f}'
                  f' v={tsm["v"][-1]:.4f} a={tsm["a"][-1]:.4f}' if tsm["lat"] else ''))
        log(msg, log_file)

        if e % 10 == 0:
            predictor.save(os.path.join(save_dir, f'predictor_{e}.pth'))

    predictor.save(os.path.join(save_dir, 'predictor_final.pth'))
    log('Predictor 训练完成', log_file)

    # ── 第二阶段：训练 controller ───────────────────────────────────────────
    log(f'\n=== 训练 Controller（{cfg.controller_epochs} epochs）===', log_file)
    log(f'参数量: {controller.total_params}', log_file)

    for e in range(1, cfg.controller_epochs + 1):
        t0 = time.time()
        epoch_end = False
        while not epoch_end:
            epoch_end, metrics, tsm = controller.update(container, predictor)
        elapsed = time.time() - t0

        now = datetime.now().strftime('%H:%M:%S')
        msg = (f'[{now}] Epoch {e:4d} ({elapsed:.1f}s) | track={metrics["track_loss"]:.4f} '
               f'smooth={metrics["smooth_loss"]:.4f} '
               f'total={metrics["total_loss"]:.4f}'
               + (f' | dv={tsm["dv"][-1]:.4f} da={tsm["da"][-1]:.4f} dkappa={tsm["dkappa"][-1]:.4f}' if tsm["dv"] else ''))
        log(msg, log_file)

        if e % 10 == 0:
            controller.save(os.path.join(save_dir, f'controller_{e}.pth'))

    controller.save(os.path.join(save_dir, 'controller_final.pth'))
    log('Controller 训练完成', log_file)


if __name__ == '__main__':
    train()
