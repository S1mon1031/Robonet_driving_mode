# MiningTruckTrack

矿车轨迹跟踪模型，基于闭环动力学学习（Predictor）+ 轨迹调整策略（Controller）。

在给定坡度和负载的条件下，学习"下发期望轨迹后矿车实际走出什么轨迹"，并在此基础上训练调整策略来补偿跟踪误差。

---

## 目录结构

```
MiningTruckTrack/
├── config_2s.yaml                 # 2s 轨迹配置（horizon=9）
├── config_10s.yaml                # 10s 轨迹配置（horizon=49，推荐）
├── network_model/
│   ├── predictor.py               # 闭环动力学模型
│   ├── controller.py              # 轨迹调整策略
│   ├── network.py                 # MLP 网络结构
│   └── parser.py                  # 配置文件解析
├── offline_train/
│   ├── data_process.py            # CSV → 训练 tensor
│   ├── train.py                   # 训练入口
│   ├── validate_controller.py     # 对比有无 Controller 的效果
│   └── container.py               # 数据加载器（支持多目录）
├── tools/
│   ├── collect_auto_driving_data.py  # 从 bag 文件采集数据
│   ├── plot_trajectory.py            # 可视化期望轨迹与响应轨迹
│   └── plot_prediction.py            # 对比实际响应轨迹与模型预测轨迹
├── data/                          # 训练 tensor（由 data_process.py 生成）
│   ├── 10s/part1/                 # 按轨迹时长和分片组织
│   └── 10s/part2/
├── state_dict/                    # 模型权重（由 train.py 生成）
│   └── 10s/                       # 按配置自动分目录
└── logs/                          # 训练日志
    └── 10s/
```

---

## 依赖

```bash
pip install -r requirements.txt
```

---

## 完整流程

### 第一步：采集数据

从自动驾驶 bag 文件中提取规划、定位、底盘、控制数据：

```bash
# 默认 5s 轨迹
python3 tools/collect_auto_driving_data.py \
    -b /path/to/record_file \
    -o /path/to/output/lap1.csv

# 10s 轨迹（推荐，训练效果更好）
python3 tools/collect_auto_driving_data.py \
    -b /path/to/record_file \
    -o /path/to/output/lap1_10s.csv \
    --traj_duration 10.0
```

每个 bag 生成一个 CSV，建议**每圈单独保存一个文件**，以保持时间连续性。

CSV 列数 = 17（固定字段）+ N×6（期望轨迹）+ N×5（响应轨迹），其中 N = traj_duration / 0.1：
- `traj_duration=2.0` → N=20 → 234 列
- `traj_duration=10.0` → N=100 → 1117 列

---

### 第二步：处理数据

将 CSV 转换为训练用 tensor：

```bash
cd /apollo/modules/MiningTruckTrack

# 10s CSV → 10s tensor（horizon 需与 config 一致）
python3 -m offline_train.data_process \
    --csv /path/to/lap1_10s.csv \
    --outdir data/10s/part1/ \
    --csv_traj_points 100 \
    --traj_points 100 \
    --horizon 49

# 2s CSV → 2s tensor
python3 -m offline_train.data_process \
    --csv /path/to/lap1_2s.csv \
    --outdir data/2s/ \
    --csv_traj_points 20 \
    --traj_points 20 \
    --horizon 9

# 旧 2s CSV 拼接成 5s tensor（无需重新采集）
python3 -m offline_train.data_process \
    --csv /path/to/lap1_2s.csv \
    --outdir data/5s/ \
    --csv_traj_points 20 \
    --traj_points 50 \
    --horizon 20
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--csv_traj_points` | CSV 每行的轨迹点数，与采集时的 `--traj_duration` 对应（2s→20，5s→50，10s→100） |
| `--traj_points` | 训练用轨迹点数，`== csv_traj_points` 为直接模式，`> csv_ttp` 为拼接模式 |
| `--horizon` | 滑动窗口半径，**必须与 config.yaml 的 horizon 一致** |

输出文件（以 10s 为例，horizon=49）：

| 文件 | 形状 | 内容 |
|------|------|------|
| `state.pt` | (N, 99, 6) | 状态序列：lateral_error, s_error, heading_error, v, a, kappa |
| `target.pt` | (N, 147, 3) | 期望目标序列：v_ref, a_ref, kappa_ref |
| `real_target.pt` | (N, 147, 3) | 同 target |
| `context.pt` | (N, 2) | 上下文：pitch（坡度，rad），load（0=轻载，1=重载） |
| `norm_stats.pt` | dict | 归一化参数，推理时使用 |

形状规律：`state=(N, 2H+1, 6)`，`target=(N, 3H, 3)`，其中 H = horizon。

**内存参考（horizon=49）：**
- 每 10 万条样本约占内存 2.3GB
- 建议单次处理不超过 25 万条样本（约 6GB）

---

### 第三步：训练

```bash
cd /apollo/modules/MiningTruckTrack

# 10s 配置从头训练
python3 -m offline_train.train \
    --config config_10s.yaml \
    --data_dir data/10s/part1

# 多个 part 合并训练
python3 -m offline_train.train \
    --config config_10s.yaml \
    --data_dir data/10s/part1 data/10s/part2 data/10s/part3

# 跳过 predictor，只训练 controller（需先将 predictor_epochs 设为 0）
python3 -m offline_train.train \
    --config config_10s.yaml \
    --data_dir data/10s/part1 \
    --resume_predictor state_dict/10s/predictor_final.pth
```

训练分两个阶段：

1. **Predictor**：学习矿车闭环动力学，`predictor_epochs` 控制轮数
2. **Controller**：固定 Predictor，学习轨迹调整策略，`controller_epochs` 控制轮数

权重保存路径由 config 中的 `save_dir` 指定（如 `state_dict/10s/`），每 10 epoch 保存一次快照，训练结束保存 `_final.pth`。

训练日志格式：

```
# Predictor
Epoch   30 | loss=0.0173 | grad=3.994 | lat=0.2630 s=0.0743 v=0.0798 a=0.0243

# Controller
Epoch   30 | track=0.xxxx smooth=0.xxxx total=0.xxxx | dv=0.xxx da=0.xxx dkappa=0.xxx
```

MAE 换算为真实值：`real_MAE = norm_MAE × (xmax - xmin) / 2`

---

### 第四步：验证 Controller 效果

```bash
# 先处理验证集 CSV
python3 -m offline_train.data_process \
    --csv /path/to/val_10s.csv \
    --outdir data/10s/val/ \
    --csv_traj_points 100 --traj_points 100 --horizon 49

# 对比有无 Controller 的轨迹跟踪误差
python3 -m offline_train.validate_controller \
    --data data/10s/val/ \
    --predictor state_dict/10s/predictor_final.pth \
    --controller state_dict/10s/controller_final.pth

# 同时保存逐步 MAE 对比
python3 -m offline_train.validate_controller \
    --data data/10s/val/ \
    --predictor state_dict/10s/predictor_final.pth \
    --controller state_dict/10s/controller_final.pth \
    --save logs/10s/val_controller.csv
```

---

## 配置文件说明

目前维护两套配置：

| 配置 | horizon | traj_points | controller_train_horizon | 适用场景 |
|------|---------|-------------|--------------------------|---------|
| `config_2s.yaml` | 9 | 20 | 9 | 快速验证 |
| `config_10s.yaml` | 49 | 100 | 30 | 正式训练（推荐） |

**关键约束：**

```
controller_train_horizon ≤ horizon
train_horizon            ≤ horizon
有效样本数/轨迹 = traj_points - 2 × horizon  （需 > 0）
```

**重要参数：**

```yaml
horizon: 49              # 滑动窗口半径，改动后必须重新处理数据
train_horizon: 49        # predictor 训练 rollout 步数（步数越长越准但越慢）
controller_train_horizon: 30  # controller rollout 步数（建议 horizon 的 60%）
predictor_epochs: 0      # 设为 0 可跳过 predictor 训练（配合 --resume_predictor）
save_dir: state_dict/10s # 权重保存目录
log_dir: logs/10s        # 日志保存目录
```

---

## 模型说明

### Predictor（闭环动力学模型）

**输入**（以 horizon=49 为例，展平后 455 维）：
- 历史 H 步 state：49 × 6 = 294 维
- 当前 state：6 维
- 历史 H 步 target：49 × 3 = 147 维
- 当前 target + 下一步 target：2 × 3 = 6 维
- context（pitch, load）：2 维

**输出**：delta_state（6 维），next_state = state + delta_state

**训练方式**：闭环滚动展开 `train_horizon` 步，用加权 MSE 损失优化，discount 衰减远期误差权重

### Controller（轨迹调整策略）

**输入**（以 horizon=49 为例）：
- 当前 state + 历史 H 步 state
- 当前 target + 历史 H 步 target
- 未来 H 步 real_target（planning 原始轨迹）
- context（pitch, load）

**输出**：`controller_train_horizon` 步的轨迹调整量 delta_target（每步 3 维：dv, da, dkappa）

**训练方式**：固定 Predictor 权重，用调整后的 target 滚动预测 `controller_train_horizon` 步，以预测轨迹与原始 planning 轨迹的误差为 loss

---

## 归一化范围

| 字段 | 范围 | 单位 |
|------|------|------|
| lateral_error | ±1.0 | m |
| s_error | ±5.0 | m |
| heading_error | ±10°（±0.1745 rad） | rad |
| v | -3 ~ 11 | m/s |
| a | ±3.0 | m/s² |
| kappa | ±0.1 | 1/m |
| pitch | ±8°（±0.1396 rad） | rad |

---

## 训练思路

### 为什么用 10s 轨迹（horizon=49）

早期使用 2s 轨迹（horizon=9）训练 Controller 时发现几乎无法收敛，根本原因是：

矿车惯性大，速度对控制指令的响应延迟约 2~5s。在 2s 的短窗口内，调整 v_ref 后车辆速度几乎没有变化，导致梯度 ∂pred_v/∂v_ref ≈ 0，Controller 无法通过 Predictor 学到有效的调整策略。

切换到 10s 轨迹（horizon=49，train_horizon=49）后，单条 rollout 覆盖 4.9s，完整包含一次加减速过程，Predictor 能观察到调整 target 后车辆速度的实际变化，梯度信号显著增强。

```
2s 窗口: target 变化 → 车速几乎不变 → 梯度消失 → Controller 无法学习
10s 窗口: target 变化 → 4.9s 后车速明显响应 → 梯度有效 → Controller 可收敛
```

### 每条训练样本的含义

以 horizon=49、一行 10s CSV 为例（CSV 采集时刻 = T）：

```
CSV 每行内容:
  states[0..99]   = 响应轨迹各步的 (lat_err, s_err, head_err, v, a, kappa)
                    对应时刻 T+0.1s ~ T+10.0s
  targets[0..99]  = 期望轨迹各步的 (v_ref, a_ref, kappa_ref)
                    对应时刻 T+0.1s ~ T+10.0s
  context         = (pitch, load)，该帧固定值
```

`build_sequences` 以 j=49 为"当前时刻"切出训练样本：

```
prev_s      = states[0:49]    历史 4.9s 的实际响应  (对应 T+0.1s ~ T+4.9s)
state       = states[49]      当前时刻实际状态       (对应 T+5.0s)
prev_t      = targets[0:49]   历史 4.9s 的期望指令  (对应 T+0.1s ~ T+4.9s)
target      = targets[49]     当前时刻期望指令       (对应 T+5.0s)
next_target = targets[50]     下一步期望指令         (对应 T+5.1s)
```

Predictor 在此基础上闭环展开 49 步（4.9s），每步用预测值替换真实值作为下一步输入，监督标签为 `states[50..98]`（真实响应）。

### Predictor 训练要点

- **闭环展开而非开环**：训练时用预测值滚动（不用真实值），使模型学到多步累积误差不发散的动力学，而非单步拟合
- **discount 衰减**：远期步骤误差权重乘以 `0.95^k`，避免长序列末端噪声主导梯度
- **加权 MSE**：v 和 s 权重高（`k_v=3.0, k_s=1.5`），lat/head/kappa 权重低，因为 Controller 主要调整纵向跟踪
- **收敛参考**：loss 从 ~0.035 降至 ~0.017 为正常，归一化空间下 v 的 MAE ≈ 0.08 对应真实 ~0.56 m/s

### Controller 训练要点

- **固定 Predictor**：Controller 训练时 Predictor 权重冻结，Controller 只优化 delta_target
- **一次输出所有步**：Controller 一次前向传播输出 `controller_train_horizon` 步（30步=3s）的全部调整量，而不是逐步输出
- **训练信号来源**：Controller 调整后的 target → Predictor 预测 next_state → 与 real_target 对比计算 loss，梯度通过 Predictor 反传到 Controller
- **controller_train_horizon 选择**：建议设为 horizon 的 60%（约 3s），覆盖完整的速度响应周期，同时避免过长导致梯度消失
- **smooth_loss**：相邻步 delta 的平方差，防止 Controller 输出的调整量在相邻时步间剧烈跳变

### 数据量与内存

| horizon | traj_points | 样本数/轨迹 | 单圈约含样本 |
|---------|-------------|------------|------------|
| 9 | 20 | 2 | 约 2k |
| 49 | 100 | 2 | 约 2k |
| 9 | 100 | 82 | 约 82k |

horizon=49 时每条轨迹只产生 2 个样本（`traj_points - 2×horizon = 100 - 98 = 2`），内存友好但需要更多圈数的数据。单 part 建议控制在 25 万样本以内（约 6GB）。

---

## 注意事项

- `--horizon` 参数必须与 config.yaml 的 `horizon` 一致，修改后需重新处理数据
- `controller_train_horizon` 和 `train_horizon` 上限均为 `horizon`，超出会 IndexError
- N 档（gear=0）数据在处理时自动跳过
- CSV 采集时建议每圈单独保存，避免跨圈拼接污染时间窗口
- 数据量大时建议分 part 处理，单 part 内存控制在 6GB 以内
- 验证集应使用训练集以外的 CSV，建议按圈或按日期划分
