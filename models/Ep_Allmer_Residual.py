import torch
import torch.nn as nn
from .utilities.match_list_lengths import match_list_lengths

class Ep_Allmer_Residual(nn.Module):
    def __init__(self, input_shape, output_shape, bottleneck=8, activation_fn=nn.ReLU):
        super(Ep_Allmer_Residual, self).__init__()
        output_len, num_tasks = output_shape
        self.conv1 = nn.Conv1d(in_channels=(num_ep_features + num_seq_features), out_channels=192, kernel_size=19, padding='same')
        self.bn1 = nn.BatchNorm1d(192)
        self.activation1 = activation_fn()
        # Assuming the residual_block function is replaced by the ResidualBlock class.
        self.res_block1 = ResidualBlock(192, 192, 3, activation_fn(), 5)
        
        # Continue with the rest of the layers following the pattern above
        # Note: This is a simplified conversion. You will need to adjust according to your residual block implementation.
        
        # Final layers
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(512, 512)  # Assuming the correct size is calculated based on your model's architecture
        self.bn2 = nn.BatchNorm1d(512)
        self.activation2 = activation_fn()
        
        # Output layer
        self.dense2 = nn.Linear(512, output_len * bottleneck)
        self.dense3 = nn.Linear(output_len * bottleneck, num_tasks)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.res_block1(x)
        # Continue with the rest of the model following the pattern above
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn2(x)
        x = self.activation2(x)
        
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.softplus(x)
        return x
