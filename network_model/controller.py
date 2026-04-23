import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from network_model.network import mlp, mlp_norm
from network_model.weight_init import weight_init
from network_model.predictor import Predictor
from offline_train.container import Container


class Controller(nn.Module):
    """
    轨迹调整策略：一次性输出1s内（train_horizon步）所有时间步的target调整量
    输入: 历史H步state + 当前state + 历史H步target + 当前target + 未来H步real_target + context
    输出: delta_targets，shape (batch, train_horizon, target_dim)
         adjusted_target[k] = real_target[k] + delta_targets[k]
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.device)
        self.horizon = cfg.horizon
        self.train_horizon = getattr(cfg, 'controller_train_horizon', cfg.train_horizon)
        self.discount = cfg.controller_discount
        self.k_smooth = cfg.controller_smooth_k
        self.max_range = cfg.controller_max_range
        self.k_stable = cfg.controller_stable_k
        self.k_lateral = cfg.controller_k_lateral
        self.k_s       = cfg.controller_k_s
        self.k_heading = cfg.controller_k_heading
        self.k_v       = cfg.controller_k_v
        self.k_a       = cfg.controller_k_a
        self.k_kappa   = cfg.controller_k_kappa
        self.iteration = 0

        # 输入维度: (H+1)*state_dim + (2H+1)*target_dim + context_dim
        controller_dim = ((cfg.horizon + 1) * cfg.state_dim
                          + (2 * cfg.horizon + 1) * cfg.target_dim
                          + cfg.context_dim)
        # 输出维度: train_horizon * target_dim（1s内所有步的delta）
        output_dim = self.train_horizon * cfg.target_dim

        if cfg.controller_type == 'mlp':
            self._controller = mlp(controller_dim,
                                   cfg.controller_hidden_depth * [cfg.controller_hidden_dim],
                                   output_dim, tanh_out=True)
        elif cfg.controller_type == 'mlp_norm':
            self._controller = mlp_norm(controller_dim,
                                        cfg.controller_hidden_depth * [cfg.controller_hidden_dim],
                                        output_dim, cfg.controller_dropout, tanh_out=True)
        else:
            raise ValueError(f'Unknown controller type: {cfg.controller_type}')

        self.apply(weight_init)
        self._controller = self._controller.to(self.device)
        self.optim = torch.optim.Adam(self.parameters(), lr=cfg.controller_lr)
        self.scheduler = None
        if cfg.controller_T_max > 0:
            self.scheduler = CosineAnnealingLR(self.optim, T_max=cfg.controller_T_max)
        self.max_norm = cfg.controller_max_norm
        self.eval()

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def control(self, state, previous_state, target, previous_target, future_real_target, context):
        """
        一次性输出1s内所有步的delta_target
        Args:
            state:              (batch, state_dim)
            previous_state:     (batch, horizon, state_dim)
            target:             (batch, target_dim)
            previous_target:    (batch, horizon, target_dim)
            future_real_target: (batch, horizon, target_dim)  未来H步planning目标
            context:            (batch, context_dim)
        Returns:
            delta_targets: (batch, train_horizon, target_dim)
        """
        prev_s = previous_state.view(previous_state.shape[0], -1)
        prev_t = previous_target.view(previous_target.shape[0], -1)
        fut_t  = future_real_target.reshape(future_real_target.shape[0], -1)

        x = torch.cat([prev_s, state, prev_t, target, fut_t, context], dim=-1)
        out = self.max_range * self._controller(x)  # (batch, train_horizon * target_dim)
        delta_targets = out.view(out.shape[0], self.train_horizon, -1)  # (batch, train_horizon, target_dim)
        return delta_targets

    def compute_loss(self, next_state_pred, next_real_target):
        """
        跟踪损失：理想状态 = (0, 0, 0, v_ref, a_ref, kappa_ref)
        next_real_target: (v_ref, a_ref, kappa_ref)
        """
        lat_loss   = F.mse_loss(next_state_pred[:, 0], torch.zeros_like(next_state_pred[:, 0]))
        s_loss     = F.mse_loss(next_state_pred[:, 1], torch.zeros_like(next_state_pred[:, 1]))
        head_loss  = F.mse_loss(next_state_pred[:, 2], torch.zeros_like(next_state_pred[:, 2]))
        v_loss     = F.mse_loss(next_state_pred[:, 3], next_real_target[:, 0])
        a_loss     = F.mse_loss(next_state_pred[:, 4], next_real_target[:, 1])
        kappa_loss = F.mse_loss(next_state_pred[:, 5], next_real_target[:, 2])

        loss = (self.k_lateral * lat_loss + self.k_s * s_loss + self.k_heading * head_loss
                + self.k_v * v_loss + self.k_a * a_loss + self.k_kappa * kappa_loss)
        mae  = torch.mean(torch.abs(next_state_pred.detach()), dim=0)
        return loss, mae

    def sequence_update(self, previous, new):
        updated = torch.empty_like(previous)
        updated[:, :-1, :] = previous[:, 1:, :].clone()
        updated[:, -1, :]  = new.clone()
        return updated

    def update(self, container: Container, predictor: Predictor, base_controller=None):
        state_seq, target_seq, real_target_seq, context_seq, epoch_end = container.sample()
        state_seq       = state_seq.to(self.device).requires_grad_(False)
        target_seq      = target_seq.to(self.device).requires_grad_(False)
        real_target_seq = real_target_seq.to(self.device).requires_grad_(False)
        context_seq     = context_seq.to(self.device).requires_grad_(False)

        self.optim.zero_grad(set_to_none=True)
        self.train()
        for p in predictor.parameters():
            p.requires_grad = False

        state           = state_seq[:, self.horizon, :]
        previous_state  = state_seq[:, :self.horizon, :]
        target          = target_seq[:, self.horizon, :]
        previous_target = target_seq[:, :self.horizon, :]
        future_real_target = real_target_seq[:, self.horizon:(self.horizon * 2), :]

        track_loss = torch.tensor(0.0, device=self.device)
        smooth_loss = torch.tensor(0.0, device=self.device)
        stable_loss = torch.tensor(0.0, device=self.device)
        time_step_metrics = {'dv': [], 'da': [], 'dkappa': [], 'mae_avg': []}

        # 一次性输出1s内所有步的delta
        delta_targets = self.control(
            state, previous_state, target, previous_target, future_real_target, context_seq)
        # delta_targets: (batch, train_horizon, target_dim)

        if self.k_stable != 0 and base_controller is not None:
            base_delta_targets = base_controller.control(
                state, previous_state, target, previous_target, future_real_target, context_seq)
            stable_loss = F.mse_loss(delta_targets, base_delta_targets.detach()) * self.k_stable

        for k in range(self.train_horizon):
            future_real_target = self.sequence_update(
                future_real_target, real_target_seq[:, self.horizon * 2 + k, :])

            delta = delta_targets[:, k, :]
            next_real_target = future_real_target[:, 0, :]
            next_target = next_real_target + delta

            next_state, _, _hx = predictor.predict(
                state, previous_state, target, previous_target, next_target, context_seq)

            loss, mae = self.compute_loss(next_state, next_real_target)
            track_loss = track_loss + (self.discount ** k) * loss

            # 相邻步delta平滑loss
            if k > 0:
                smooth_loss = smooth_loss + F.mse_loss(
                    delta_targets[:, k, :], delta_targets[:, k - 1, :]) * self.k_smooth

            if epoch_end:
                time_step_metrics['dv'].append(float(delta[:, 0].abs().mean()))
                time_step_metrics['da'].append(float(delta[:, 1].abs().mean()))
                time_step_metrics['dkappa'].append(float(delta[:, 2].abs().mean()))
                time_step_metrics['mae_avg'].append(float(mae.mean()))

            previous_state  = self.sequence_update(previous_state, state)
            previous_target = self.sequence_update(previous_target, target)
            state  = next_state
            target = next_target

        total_loss = (track_loss + smooth_loss + stable_loss) / self.train_horizon
        total_loss.backward()
        if self.max_norm != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_norm)
        else:
            grad_norm = 0
        self.optim.step()
        if self.scheduler:
            self.scheduler.step()

        self.eval()
        for p in predictor.parameters():
            p.requires_grad = True
        self.iteration += 1

        metrics = {
            'track_loss':  float(track_loss.item() / self.train_horizon),
            'smooth_loss': float(smooth_loss.item() / self.train_horizon),
            'total_loss':  float(total_loss.item()),
            'grad_norm':   float(grad_norm),
        }
        return epoch_end, metrics, time_step_metrics

    def save(self, fp):
        torch.save({'controller': self.state_dict()}, fp)

    def load(self, fp):
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, weights_only=True)
        self.load_state_dict(state_dict['controller'])
