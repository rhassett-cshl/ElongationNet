import torch
import torch.nn as nn

class Ep_Allmer_LSTM(nn.Module):
    def __init__(self, input_size):
        super(Ep_Allmer_LSTM, self).__init__()
        self.name = "lstm"
        layer_size = 16
        num_layers=1
        bidirectional=False
        self.lstm = nn.LSTM(input_size, layer_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.bidirectional_linear = nn.Linear(2 * layer_size, 1)
        self.linear = nn.Linear(layer_size, 1)
        self.bidirectional = bidirectional

    def forward(self, Y_ji, N_ji):
        x = torch.cat((Y_ji, N_ji), axis=-1)
        x, _ = self.lstm(x)
        if self.bidirectional:
            x = self.bidirectional_linear(x)
        else:
            x = self.linear(x)
        return x.squeeze(-1)