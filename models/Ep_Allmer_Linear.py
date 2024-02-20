import torch
import torch.nn as nn

class Ep_Allmer_Linear(nn.Module):
    def __init__(self, num_ep_features, num_seq_features):
        super(Ep_Allmer_Linear, self).__init__()
        self.name = "ep_seq_linear"
        self.y_linear = nn.Linear(num_ep_features, 1)
        self.n_linear = nn.Linear(num_seq_features, 1)
        self.final_linear = nn.Linear(2, 1)

    def forward(self, Y_ji, N_ji):
        y = self.y_linear(Y_ji)
        n = self.n_linear(N_ji)
        x = torch.cat((y, n), axis=-1)
        x = self.final_linear(x)
        return x.squeeze(-1)