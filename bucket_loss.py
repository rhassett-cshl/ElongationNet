import torch
import torch.nn as nn

class BucketLoss(nn.Module):
    def __init__(self):
        super(BucketLoss, self).__init__()

    def forward(self, X_ji, C_j, rho_ji, mask):
        loss = (X_ji * rho_ji + C_j * torch.exp(-rho_ji) - X_ji * torch.log(C_j)) * mask
                
        non_padded_elements = mask.sum()
        
        avg_loss = loss.sum() / non_padded_elements
                    
        return avg_loss
