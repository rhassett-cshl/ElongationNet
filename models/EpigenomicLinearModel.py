

class EpLinearModel(nn.Module):
	def __init__(self, input_size):
            super(EpLinearModel, self).__init__()
            self.name = "ep_linear"
            self.linear = nn.Linear(input_size, 1)

        def forward(self, Y_ji):
            x = self.linear(Y_ji)
            return x.squeeze(-1)  
