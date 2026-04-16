# MiningTruckTrack

矿车轨迹跟踪模型，基于闭环动力学学习（Predictor）+ 轨迹调整策略（Controller）。

在给定坡度和负载的条件下，学习"下发期望轨迹后矿车实际走出什么轨迹"，并在此基础上训练调整策略来补偿跟踪误差。

---

## 目录结构

```
MiningTruckTrack/
├── config.yaml                    # 模型与训练配置
├── network_model/
│   ├── predictor.py               # 闭环动力学模型
│   ├── controller.py              # 轨迹调整策略
│   ├── network.py                 # MLP 网络结构
│   └── parser.py                  # 配置文件解析
├── offline_train/
│   ├── data_process.py            # CSV → 训练 tensor
│   ├── train.py                   # 训练入口
│   ├── validate.py                # 验证脚本
│   └── container.py               # 数据加载器
├── tools/
│   ├── collect_auto_driving_data.py  # 从 bag 文件采集数据
│   ├── plot_trajectory.py            # 可视化期望轨迹与响应轨迹
│   └── plot_prediction.py            # 对比实际响应轨迹与模型预测轨迹
├── data/                          # 训练 tensor（由 data_process.py 生成）
├── state_dict/                    # 模型权重（由 train.py 生成）
└── logs/                          # 训练日志（由 train.py 生成）
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
python3 tools/collect_auto_driving_data.py \
    -b /path/to/record_file \
    -o /path/to/output/lap1.csv
```

每个 bag 生成一个 CSV，建议**每圈单独保存一个文件**，以保持时间连续性。

CSV 包含 234 列：时间戳、当前状态（位置/速度/加速度/底盘信息）、期望轨迹（20点×6维）、响应轨迹（20点×5维）。

---

### 第二步：处理数据

将 CSV 转换为训练用 tensor：

```bash
cd /apollo/modules/MiningTruckTrack

# 单个 CSV
python3 -m offline_train.data_process \
    --csv /path/to/lap1.csv \
    --outdir data/

# 多个 CSV 合并（推荐，覆盖更多场景）
python3 -m offline_train.data_process \
    --csv /path/to/*.csv \
    --outdir data/
```

输出文件保存在 `data/` 目录：

| 文件 | 形状 | 内容 |
|------|------|------|
| `state.pt` | (N, 19, 6) | 状态序列：lateral_error, s_error, heading_error, v, a, kappa |
| `target.pt` | (N, 27, 3) | 期望目标序列：v_ref, a_ref, kappa_ref |
| `real_target.pt` | (N, 27, 3) | 同 target |
| `context.pt` | (N, 2) | 上下文：pitch（坡度，rad），load（0=轻载，1=重载） |
| `norm_stats.pt` | dict | 归一化参数，推理时使用 |

归一化范围：

| 字段 | 范围 | 单位 |
|------|------|------|
| lateral_error | ±1.0 | m |
| s_error | ±5.0 | m |
| heading_error | ±10° (±0.1745 rad) | rad |
| v | -3 ~ 11 | m/s |
| a | ±3.0 | m/s² |
| kappa | ±0.1 | 1/m |
| pitch | ±8° (±0.1396 rad) | rad |

---

### 第三步：训练

```bash
cd /apollo/modules/MiningTruckTrack

# 从头训练
python3 -m offline_train.train

# 从已有权重继续训练
python3 -m offline_train.train \
    --resume_predictor state_dict/predictor_300.pth
```

训练分两个阶段，由 `config.yaml` 控制：

- **Predictor**：学习矿车动力学，建议先训练此模型
- **Controller**：基于已训练的 Predictor 学习轨迹调整策略，设 `controller_epochs: 0` 可跳过

模型权重保存在 `state_dict/`，每 50 epoch 保存一次快照，训练结束保存 `_final.pth`。

训练日志保存在 `logs/train_log_日期.txt`，格式：

```
Epoch  100 | loss=0.0039 | grad=0.666 | lat=0.0349 s=0.0120 v=0.0034 a=0.0210
```

MAE 换算为真实值：`real_MAE = norm_MAE × (xmax - xmin) / 2`

---

### 第四步：验证

```bash
cd /apollo/modules/MiningTruckTrack

python3 -m offline_train.validate \
    --csv /path/to/val/*.csv \
    --model state_dict/predictor_final.pth

# 同时保存逐步 MAE
python3 -m offline_train.validate \
    --csv /path/to/val/*.csv \
    --model state_dict/predictor_final.pth \
    --save logs/val_mae.csv
```

输出各维度逐步预测误差（真实单位），可评估闭环累积误差随预测步数的增长情况。

---

## 关键配置（config.yaml）

```yaml
device: cpu            # 有 GPU 改为 cuda

horizon: 9             # 历史/未来窗口步数，需与数据处理一致，勿随意修改
train_horizon: 9       # 训练展开步数

batch_size: 2048       # 建议：样本总数 / 500 左右
predictor_epochs: 300  # 数据量大时可适当减少
controller_epochs: 0   # 先训练 predictor，确认效果后再开启

predictor_k_v: 2.0     # 速度损失权重（较高，速度误差影响大）
predictor_k_lateral: 1.0
predictor_k_s: 1.0
```

---

## 模型说明

### Predictor（闭环动力学模型）

**输入**（展平后 95 维）：
- 历史 9 步 state + 当前 state：10 × 6 = 60 维
- 历史 9 步 target + 当前 target + 下一步 target：11 × 3 = 33 维
- context（pitch, load）：2 维

**输出**：delta_state（6 维），next_state = state + delta_state

**训练方式**：闭环滚动展开 9 步，用加权 MSE 损失优化

### Controller（轨迹调整策略）

**输入**：当前状态、历史序列、context

**输出**：1s 内的轨迹调整量 delta_target（9 步 × 3 维 = 27 维）

**训练方式**：固定 Predictor 权重，用 Controller 输出的 delta_target 调整期望轨迹后滚动预测，以实际轨迹误差为损失

---

## 注意事项

- `horizon` 和 `HORIZON`（data_process.py）必须保持一致，修改后需重新处理数据
- N 档数据在处理时自动跳过
- 验证集应使用训练集以外的 CSV，建议按圈或按日期划分
- 多个 CSV 处理时每个文件独立处理，避免不同圈之间的时间窗口污染
