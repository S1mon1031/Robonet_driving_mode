import os
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset


class TruckDataset(Dataset):
    """
    数据格式（由 data_process.py 生成）：
      state:       (N, 2*horizon+1, state_dim)   — 前H步历史 + 当前 + 后H步
      target:      (N, 3*horizon,   target_dim)  — 前H步历史 + 当前 + 后2H步
      real_target: (N, 3*horizon,   target_dim)  — 同 target（planning即期望）
      context:     (N, context_dim)              — (pitch, load)，当前时刻固定
    """

    def __init__(self, data_dir):
        self.state       = torch.load(os.path.join(data_dir, 'state.pt'),       weights_only=True)
        self.target      = torch.load(os.path.join(data_dir, 'target.pt'),      weights_only=True)
        self.real_target = torch.load(os.path.join(data_dir, 'real_target.pt'), weights_only=True)
        self.context     = torch.load(os.path.join(data_dir, 'context.pt'),     weights_only=True)
        assert self.state.shape[0] == self.target.shape[0] == self.context.shape[0]
        self.total_length = self.state.shape[0]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        return (self.state[idx], self.target[idx],
                self.real_target[idx], self.context[idx])


class Container:
    def __init__(self, data_dirs, batch_size, device=None):
        """
        data_dirs: 单个目录路径（str）或多个目录路径的列表（list[str]）
        device: 若为 cuda，数据全量预加载到 GPU，用 randperm 切片代替 DataLoader，
                避免逐条 __getitem__ 的 GPU tensor 拼接开销
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        all_state, all_target, all_real_target, all_context = [], [], [], []
        for d in data_dirs:
            ds = TruckDataset(d)
            print(f'  加载 {d}: {ds.total_length} 条样本')
            all_state.append(ds.state)
            all_target.append(ds.target)
            all_real_target.append(ds.real_target)
            all_context.append(ds.context)

        self.state       = torch.cat(all_state,       dim=0)
        self.target      = torch.cat(all_target,      dim=0)
        self.real_target = torch.cat(all_real_target, dim=0)
        self.context     = torch.cat(all_context,     dim=0)
        self.N           = self.state.shape[0]
        print(f'  合计 {self.N} 条样本')

        self.batch_size = batch_size
        self.device     = torch.device(device) if device else torch.device('cpu')

        # 预加载到目标设备
        if self.device.type != 'cpu':
            self.state       = self.state.to(self.device)
            self.target      = self.target.to(self.device)
            self.real_target = self.real_target.to(self.device)
            self.context     = self.context.to(self.device)
            print(f'  数据已预加载到 {self.device}')

        self._reset()

    def _reset(self):
        self._perm = torch.randperm(self.N, device=self.device)
        self._pos  = 0

    def sample(self):
        start = self._pos
        end   = min(start + self.batch_size, self.N)
        idx   = self._perm[start:end]

        state       = self.state[idx]
        target      = self.target[idx]
        real_target = self.real_target[idx]
        context     = self.context[idx]

        epoch_end = (end >= self.N)
        if epoch_end:
            self._reset()
        else:
            self._pos = end

        return state, target, real_target, context, epoch_end
