import torch
import torch.nn as nn
import torch.nn.functional as F

from network_model.network import mlp, mlp_norm, LSTMNet
from network_model.weight_init import weight_init
from offline_train.container import Container


class Predictor(nn.Module):
    """
    闭环动力学模型：预测下一时刻车辆状态
    state:   (lateral_error, s_error, heading_error, v, a, kappa)  6维
    target:  (v_ref, a_ref, kappa_ref)                             3维
    context: (pitch, load)                                         2维

    输入: 历史H步state + 当前state + 历史H步target + 当前target + 下一步target + context
    输出: delta_state，next_state = state + delta_state

    predictor_type:
      mlp / mlp_norm : 将所有输入展平拼接后送入 MLP
      lstm           : 历史序列 (prev_state||prev_target) 送 LSTM，
                       当前 state/target/next_target/context 作 extra 拼接解码头
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.device)
        self.horizon = cfg.horizon
        self.train_horizon = cfg.train_horizon
        self.discount = cfg.predictor_discount
        self.max_range = cfg.predictor_max_range
        self.k_lateral  = cfg.predictor_k_lateral
        self.k_s        = cfg.predictor_k_s
        self.k_heading  = cfg.predictor_k_heading
        self.k_v        = cfg.predictor_k_v
        self.k_a        = cfg.predictor_k_a
        self.k_kappa    = cfg.predictor_k_kappa
        self.k_stable = cfg.predictor_stable_k
        self.iteration = 0

        # 输入维度: (H+1)*state_dim + (H+2)*target_dim + context_dim
        predict_dim = ((cfg.horizon + 1) * cfg.state_dim
                       + (cfg.horizon + 2) * cfg.target_dim
                       + cfg.context_dim)

        if cfg.predictor_type == 'mlp':
            self._predictor = mlp(predict_dim,
                                  cfg.predictor_hidden_depth * [cfg.predictor_hidden_dim],
                                  cfg.state_dim, tanh_out=True)
        elif cfg.predictor_type == 'mlp_norm':
            self._predictor = mlp_norm(predict_dim,
                                       cfg.predictor_hidden_depth * [cfg.predictor_hidden_dim],
                                       cfg.state_dim, cfg.predictor_dropout, tanh_out=True)
        elif cfg.predictor_type == 'lstm':
            # 序列输入：历史 H 步，每步 = state(6) || target(3)
            seq_feat_dim = cfg.state_dim + cfg.target_dim
            # extra 输入：当前 state + 当前 target + 下一步 target + context
            extra_dim = cfg.state_dim + cfg.target_dim + cfg.target_dim + cfg.context_dim
            self._predictor = LSTMNet(
                seq_feat_dim  = seq_feat_dim,
                extra_dim     = extra_dim,
                lstm_hidden   = cfg.lstm_hidden_size,
                lstm_layers   = cfg.lstm_num_layers,
                mlp_dims      = cfg.lstm_head_dim,
                out_dim       = cfg.state_dim,
                dropout       = cfg.predictor_dropout,
                tanh_out      = True,
            )
        else:
            raise ValueError(f'Unknown predictor type: {cfg.predictor_type}')

        self.apply(weight_init)
        self._predictor = self._predictor.to(self.device)
        self.optim = torch.optim.Adam(self.parameters(),
                                      lr=cfg.predictor_lr,
                                      weight_decay=cfg.predictor_weight_decay)
        self.max_norm = cfg.predictor_max_norm
        self.eval()

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict(self, state, previous_state, target, previous_target, next_target, context,
                hx=None):
        """
        Args:
            state:          (batch, state_dim)
            previous_state: (batch, horizon, state_dim)  — LSTM模式rollout时可传 None
            target:         (batch, target_dim)
            previous_target:(batch, horizon, target_dim) — LSTM模式rollout时可传 None
            next_target:    (batch, target_dim)
            context:        (batch, context_dim)
            hx:             (h_n, c_n) LSTM hidden state，rollout时传入上步输出，None则从头算
        Returns:
            next_state:  (batch, state_dim)
            delta_state: (batch, state_dim)
            hx_out:      (h_n, c_n) 仅 LSTM 模式有效，MLP 模式返回 None
        """
        if isinstance(self._predictor, LSTMNet):
            extra = torch.cat([state, target, next_target, context], dim=-1)
            if hx is not None:
                # rollout：只送当前一步作为序列输入，复用 hidden state
                cur_feat = torch.cat([state, target], dim=-1).unsqueeze(1)  # (B,1,feat)
                delta_state, hx_out = self._predictor(cur_feat, extra, hx=hx)
            else:
                # 初始步：用完整历史序列初始化 hidden state
                seq = torch.cat([previous_state, previous_target], dim=-1)  # (B,H,feat)
                delta_state, hx_out = self._predictor(seq, extra, hx=None)
            delta_state = self.max_range * delta_state
        else:
            prev_s = previous_state.view(previous_state.shape[0], -1)
            prev_t = previous_target.view(previous_target.shape[0], -1)
            x = torch.cat([prev_s, state, prev_t, target, next_target, context], dim=-1)
            delta_state = self.max_range * self._predictor(x)
            hx_out = None
        next_state = state + delta_state
        return next_state, delta_state, hx_out

    def compute_loss(self, next_state_pred, next_state_real):
        """分别对六个维度加权计算MSE损失"""
        lat_loss   = F.mse_loss(next_state_pred[:, 0], next_state_real[:, 0])
        s_loss     = F.mse_loss(next_state_pred[:, 1], next_state_real[:, 1])
        head_loss  = F.mse_loss(next_state_pred[:, 2], next_state_real[:, 2])
        v_loss     = F.mse_loss(next_state_pred[:, 3], next_state_real[:, 3])
        a_loss     = F.mse_loss(next_state_pred[:, 4], next_state_real[:, 4])
        kappa_loss = F.mse_loss(next_state_pred[:, 5], next_state_real[:, 5])

        total = (self.k_lateral * lat_loss + self.k_s * s_loss
                 + self.k_heading * head_loss + self.k_v * v_loss
                 + self.k_a * a_loss + self.k_kappa * kappa_loss)
        mae = torch.mean(torch.abs(next_state_pred.detach() - next_state_real), dim=0)
        return total, mae

    def sequence_update(self, previous, new):
        """滑动窗口更新历史序列"""
        updated = torch.empty_like(previous)
        updated[:, :-1, :] = previous[:, 1:, :].clone()
        updated[:, -1, :]  = new.clone()
        return updated

    def update(self, container: Container, base_predictor=None):
        state_seq, target_seq, _, context_seq, epoch_end = container.sample()
        state_seq   = state_seq.to(self.device).requires_grad_(False)
        target_seq  = target_seq[:, :(2 * self.horizon + 1), :].to(self.device).requires_grad_(False)
        context_seq = context_seq.to(self.device).requires_grad_(False)  # (batch, context_dim)

        self.optim.zero_grad(set_to_none=True)
        self.train()

        state           = state_seq[:, self.horizon, :]
        previous_state  = state_seq[:, :self.horizon, :]
        target          = target_seq[:, self.horizon, :]
        previous_target = target_seq[:, :self.horizon, :]

        total_loss, stable_loss = 0, 0
        time_step_metrics = {'lat': [], 's': [], 'head': [], 'v': [], 'a': [], 'kappa': [], 'avg': []}

        hx = None  # LSTM hidden state，第一步从历史序列初始化，后续复用
        for k in range(self.train_horizon):
            next_target = target_seq[:, self.horizon + k + 1, :]
            next_state_pred, delta, hx = self.predict(
                state, previous_state, target, previous_target, next_target, context_seq,
                hx=hx)
            next_state_real = state_seq[:, self.horizon + k + 1, :]

            loss, mae = self.compute_loss(next_state_pred, next_state_real)
            total_loss = total_loss + (self.discount ** k) * loss

            if epoch_end:
                time_step_metrics['lat'].append(float(mae[0]))
                time_step_metrics['s'].append(float(mae[1]))
                time_step_metrics['head'].append(float(mae[2]))
                time_step_metrics['v'].append(float(mae[3]))
                time_step_metrics['a'].append(float(mae[4]))
                time_step_metrics['kappa'].append(float(mae[5]))
                time_step_metrics['avg'].append(float(mae.mean()))

            if self.k_stable != 0 and base_predictor is not None:
                _, base_delta, _ = base_predictor.predict(
                    state, previous_state, target, previous_target, next_target, context_seq)
                stable_loss = stable_loss + F.mse_loss(delta, base_delta.detach()) * self.k_stable

            previous_state  = self.sequence_update(previous_state, state)
            previous_target = self.sequence_update(previous_target, target)
            state  = next_state_pred
            target = next_target

        total_loss = total_loss / self.train_horizon + stable_loss / self.train_horizon
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        self.optim.step()

        self.eval()
        self.iteration += 1

        metrics = {
            'total_loss': float(total_loss.item()),
            'grad_norm':  float(grad_norm),
        }
        return epoch_end, metrics, time_step_metrics

    def save(self, fp):
        torch.save({'predictor': self.state_dict()}, fp)

    def load(self, fp):
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, weights_only=True)
        self.load_state_dict(state_dict['predictor'])
