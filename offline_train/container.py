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
        self.state       = torch.load(os.path.join(data_dir, 'state.pt'),       mmap=True)
        self.target      = torch.load(os.path.join(data_dir, 'target.pt'),      mmap=True)
        self.real_target = torch.load(os.path.join(data_dir, 'real_target.pt'), mmap=True)
        self.context     = torch.load(os.path.join(data_dir, 'context.pt'),     mmap=True)
        assert self.state.shape[0] == self.target.shape[0] == self.context.shape[0]
        self.total_length = self.state.shape[0]

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        return (self.state[idx], self.target[idx],
                self.real_target[idx], self.context[idx])


class Container:
    def __init__(self, data_dirs, batch_size):
        """
        data_dirs: 单个目录路径（str）或多个目录路径的列表（list[str]）
        每个目录下需包含 state.pt / target.pt / real_target.pt / context.pt
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        datasets = []
        for d in data_dirs:
            ds = TruckDataset(d)
            print(f'  加载 {d}: {ds.total_length} 条样本')
            datasets.append(ds)

        dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        print(f'  合计 {len(dataset)} 条样本')

        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.iterator   = iter(self.dataloader)
        self.num_push   = len(dataset)

    def sample(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        state, target, real_target, context = batch
        epoch_end = len(state) < self.batch_size
        return state, target, real_target, context, epoch_end
