import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, num_layers=5, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.num_layers = num_layers
        self.base_rate = 2

        self.initial_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', dilation=1)
        self.initial_bn = nn.BatchNorm1d(in_channels)
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(1, num_layers):
            dilation_rate = self.base_rate ** i
            self.conv_layers.append(nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', dilation=dilation_rate))
            self.bn_layers.append(nn.BatchNorm1d(in_channels))
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        
        out = self.initial_conv(x)
        out = self.initial_bn(out)

        for i in range(self.num_layers - 1):
            out = self.relu(out)
            out = self.dropout(out)
            out = self.conv_layers[i](out)
            out = self.bn_layers[i](out)
        
        out += residual
        
        return self.relu(out)



class Ep_Allmer_Residual(nn.Module):
    def __init__(self, num_ep_features, num_seq_features, window_size, bottleneck=8):
        super(Ep_Allmer_Residual, self).__init__()
        num_output_features = 1
        output_len = window_size
        self.output_len = output_len
        self.bottleneck = bottleneck
        
        self.conv1 = nn.Conv1d(in_channels=(num_ep_features + num_seq_features), out_channels=192, kernel_size=19, padding='same')
        self.bn1 = nn.BatchNorm1d(192)
        self.relu = nn.ReLU()
        self.res_block1 = ResidualBlock(192, 192, 3, 5, 0.1)
        self.maxpool = nn.MaxPool1d(4)
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv1d(in_channels=192, out_channels=256, kernel_size=7, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        self.res_block2 = ResidualBlock(256, 256, 3, 5, 0.1)
        
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=7, padding='same')
        self.bn3 = nn.BatchNorm1d(512)
        self.res_block3 = ResidualBlock(512, 512, 3, 4, 0.1)
        self.dropout2 = nn.Dropout(0.2)
        
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.3)
        
        self.dense2 = nn.Linear(512, output_len * bottleneck)
        self.bn5 = nn.BatchNorm1d(output_len * bottleneck)
        
        self.conv4 = nn.Conv1d(in_channels=output_len, out_channels=256, kernel_size=7, padding='same')
        self.bn6 = nn.BatchNormalization(256)
        self.res_block4 = ResidualBlock(256, 256, 3, 5, 0.1)
        
        self.dense3 = nn.Linear(256, num_output_features)
        #self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_block1(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.res_block2(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.res_block3(x)
        x = self.maxpool(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.dense2(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = x.view(-1, self.output_len, self.bottleneck)
        x = self.dropout1(x)
        
        x = self.conv4(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.res_block4(x)
        
        x = self.dense3(x)
        #x = self.softplus(x)
        
        return x
