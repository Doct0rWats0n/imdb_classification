import torch
import torch.nn as nn
from parameters import E_DIM, H_DIM


class RNN(nn.Module):

    def __init__(self, e_dim=E_DIM, h_dim=H_DIM):
        super().__init__()
        self.rnn = nn.RNN(e_dim, h_dim, batch_first=True)
        self.fc1 = nn.Linear(h_dim, h_dim * 2)

    def forward(self, x):
        x, hs = self.rnn(x)
        x = torch.cat((x, hs), dim=1)
        x = self.fc1(x)
        return x
