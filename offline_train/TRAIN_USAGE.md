# 训练使用说明

## 完整流程

### 第一步：采集数据

```bash
cd /apollo/modules/tools/control_tools
python3 collect_auto_driving_data.py \
    -b /path/to/*.record \
    -o /path/to/output/batch1.csv
```

### 第二步：处理数据（生成训练tensor）

```bash
cd /apollo/modules/MiningTruckTrack

# 单个 CSV
python3 -m offline_train.data_process --csv /path/to/batch1.csv --outdir data/

# 多个 CSV（合并为一个数据集）
python3 -m offline_train.data_process --csv /path/to/*.csv --outdir data/
```

输出文件（保存在 `data/` 目录）：

| 文件 | 形状 | 说明 |
|------|------|------|
| `state.pt` | (N, 19, 6) | 状态序列：lateral_error, s_error, heading_error, v, a, kappa |
| `target.pt` | (N, 27, 3) | 目标序列：v_ref, a_ref, kappa_ref |
| `real_target.pt` | (N, 27, 3) | 同 target（planning 期望轨迹） |
| `context.pt` | (N, 2) | 上下文：pitch, load |
| `norm_stats.pt` | dict | 归一化参数，推理时使用 |

### 第三步：训练

```bash
cd /apollo/modules/MiningTruckTrack

# 从头训练
python3 -m offline_train.train

# 从已有权重继续训练
python3 -m offline_train.train --resume_predictor state_dict/predictor_300.pth

# 同时加载 predictor 和 controller
python3 -m offline_train.train --resume_predictor state_dict/predictor_final.pth --resume_controller state_dict/controller_final.pth
```

---

## 训练阶段

训练分两个阶段，由 `config.yaml` 控制：

**第一阶段：Predictor（闭环动力学模型）**
- 学习"给定期望轨迹和上下文，矿车实际走出什么轨迹"
- 权重保存：每 50 epoch 存一次 + 最终 `predictor_final.pth`

**第二阶段：Controller（轨迹调整策略）**
- 依赖训练好的 predictor，输出 1s 内的轨迹调整量
- 设 `controller_epochs: 0` 可跳过此阶段

---

## 输出文件

模型权重保存在 `state_dict/` 目录：

```
state_dict/
├── predictor_50.pth
├── predictor_100.pth
├── ...
├── predictor_final.pth
├── controller_50.pth
└── controller_final.pth
```

---

## 训练日志说明

```
Epoch  100 | loss=0.0039 | grad=0.666 | lat=0.0349 s=0.0120 v=0.0034 a=0.0210
```

| 字段 | 说明 | 换算系数 | 真实单位 |
|------|------|---------|---------|
| `loss` | 加权 MSE 总损失 | — | — |
| `grad` | 梯度裁剪前的范数 | — | — |
| `lat` | 横向误差 MAE（归一化） | ×1.0 | m |
| `s` | 纵向误差 MAE（归一化） | ×5.0 | m |
| `v` | 速度 MAE（归一化） | ×7.0 | m/s |
| `a` | 加速度 MAE（归一化） | ×3.0 | m/s² |

> 换算公式：real_MAE = norm_MAE × (xmax - xmin) / 2
> 例：lat MAE=0.035 → 真实横向误差约 **0.035m**

---

## 关键配置（config.yaml）

```yaml
horizon: 9          # 历史/未来窗口步数，需与数据处理一致
train_horizon: 9    # 训练展开步数
batch_size: 2048    # 推荐：样本数 / 500 左右
predictor_epochs: 300
controller_epochs: 0   # 先训练 predictor，确认效果后再开启
device: cpu            # 有 GPU 改为 cuda
```
