import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, n_class=1):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)

    def forward(self, x):  # x shape: (batch_size, seq_len, input_size)
        out, _ = self.cell(x)
        out = out.reshape(-1, self.hidden_dim)
        out = self.linear(out)  # out shape: (batch_size, n_class=1)
        return out
