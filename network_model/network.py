import torch
import torch.nn as nn


"""
    Network architecture of ErrorTrack.
    Adapted from https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
"""


class NormedLinear(nn.Linear):
    """
    Linear layer with optionally dropout, LayerNorm, and activation.
    """

    def __init__(self, *args, dropout=0., act, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x)) if self.act is not None else self.ln(x)
        # return self.act(x) if self.act is not None else self.ln(x)

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"bias={self.bias is not None}{repr_dropout}, " \
               f"act={self.act.__class__.__name__})"


def mlp_norm(in_dim, mlp_dims, out_dim, dropout=0., tanh_out=False):
    """
    input -> NormedLinear(Linear -> Dropout -> LayerNorm -> act)
          -> NormedLinear(Linear -> LayerNorm -> act) -> ...
          -> NormedLinear(Linear -> LayerNorm -> act)
          -> NormedLinear(Linear -> LayerNorm)/Linear -> output
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    net = nn.ModuleList()
    for i in range(len(dims) - 2):
        net.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0), act=nn.ELU()))
    # net.append(NormedLinear(dims[-2], dims[-1], act=None))
    net.append(nn.Linear(dims[-2], dims[-1]))  # no layer norm
    if tanh_out:
        net.append(nn.Tanh())
    return nn.Sequential(*net)


def mlp(in_dim, mlp_dims, out_dim, tanh_out=False):
    """
    input -> (Linear -> ELU) -> ... -> (Linear -> ELU) -> output
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]

    net = nn.ModuleList()
    for i in range(len(dims) - 2):
        net.append(nn.Linear(dims[i], dims[i + 1]))
        net.append(nn.ELU())  # net.append(nn.Mish())  #
    net.append(nn.Linear(dims[-2], dims[-1]))

    if tanh_out:
        net.append(nn.Tanh())

    return nn.Sequential(*net)


class LSTMNetPlus(nn.Module):
    """
    与参考版本（0415）完全一致的 LSTM 实现。
    Linear → [Dropout] → LayerNorm → ELU 解码头，无 hx 传递，无 input_norm。
    """

    def __init__(self, seq_feat_dim, extra_dim, lstm_hidden, lstm_layers,
                 mlp_dims, out_dim, dropout=0.0, tanh_out=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        mlp_in = lstm_hidden + extra_dim
        if isinstance(mlp_dims, int):
            mlp_dims = [mlp_dims]
        dims = [mlp_in] + mlp_dims + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if dropout > 0 and i == 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if tanh_out:
            layers.append(nn.Tanh())
        self.head = nn.Sequential(*layers)

    def forward(self, seq_input, extra_input=None):
        _, (h_n, _) = self.lstm(seq_input)
        h_last = h_n[-1]
        if extra_input is not None:
            x = torch.cat([h_last, extra_input], dim=-1)
        else:
            x = h_last
        return self.head(x)


class LSTMNet(nn.Module):
    """
    LSTM 编码器 + MLP 解码头。

    输入分两路：
      seq_input  : (batch, seq_len, seq_feat_dim)  — 时序部分（历史状态+历史目标）
      extra_input: (batch, extra_dim)              — 非时序部分（当前状态/目标/context）

    内部流程：
      1. LSTM 处理 seq_input，取最后时刻隐状态 h_n[-1]
      2. 将 h_n 与 extra_input 拼接
      3. MLP 解码头输出结果
    """

    def __init__(self, seq_feat_dim, extra_dim, lstm_hidden, lstm_layers,
                 mlp_dims, out_dim, dropout=0.0, tanh_out=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        mlp_in = lstm_hidden + extra_dim
        if isinstance(mlp_dims, int):
            mlp_dims = [mlp_dims]
        dims = [mlp_in] + mlp_dims + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if dropout > 0 and i == 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ELU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if tanh_out:
            layers.append(nn.Tanh())
        self.head = nn.Sequential(*layers)

    def forward(self, seq_input, extra_input=None, hx=None):
        """
        seq_input  : (batch, seq_len, seq_feat_dim)
        extra_input: (batch, extra_dim) 或 None
        hx         : (h_0, c_0) 或 None，用于 rollout 时传入上一步 hidden state
        返回        : output (batch, out_dim), (h_n, c_n)
        """
        _, (h_n, c_n) = self.lstm(seq_input, hx)  # h_n: (num_layers, batch, hidden)
        h_last = h_n[-1]                            # (batch, hidden)
        if extra_input is not None:
            x = torch.cat([h_last, extra_input], dim=-1)
        else:
            x = h_last
        return self.head(x), (h_n, c_n)
