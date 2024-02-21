import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, X_ji, C_j, rho_ji):
        term1 = X_ji * rho_ji
        term2 = C_j * torch.exp(-rho_ji)
        term3 = X_ji * torch.log(C_j)
        loss = term1 + term2 - term3
        
        # calculate average loss per site
        return (loss).mean()