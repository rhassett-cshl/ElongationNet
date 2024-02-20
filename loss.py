import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, X_ji, C_j, rho_ji):
        large_rho = X_ji * rho_ji
        small_rho = C_j * torch.exp(-rho_ji)
        constant_term = X_ji * torch.log(C_j)
        loss = large_rho + small_rho - constant_term
        #print(f"Large:{large_rho.mean()}\nSmall:{small_rho.mean()}\nConstant:{constant_term.mean()}")
        
        # calculate average loss per site
        return (loss).mean()