#!/usr/bin/env python
# coding: utf-8

cell_type = "k562"

""" Configure path to config file and select whether hyperparameter sweeping or not """
config_folder_path = "./configs/"
config_file_name = "cnn_2"
results_folder_path = "./results/"
sweep_config = {
    'method': 'grid'
}
is_sweeping=False

input_data_file = f'./data/{cell_type}_datasets.pkl'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math 
import json
import pyBigWig
import torch
import torch.utils.data as td
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset, BatchSampler


""" Load datasets """
with open(input_data_file, 'rb') as file:
    combined_datasets = pickle.load(file)
    
nucleotides = ['A', 'T', 'G', 'C']

train_data = combined_datasets['train']
valid_data = combined_datasets['valid']
test_data = combined_datasets['test']

print(train_data.iloc[0])

column_names = np.array(train_data.columns)
feature_names = column_names[6:16]
num_features = len(feature_names)
print(feature_names)
num_samples = train_data.shape[0]
num_seq_features = len(nucleotides)

print("Number of Samples: " + str(num_samples))
print("Number of Features: " + str(num_features))

cuda_available = torch.cuda.is_available()
print("CUDA (GPU support) is available:", cuda_available)
num_gpus = torch.cuda.device_count()
print("Number of GPUs available:", num_gpus)

torch.backends.cudnn.benchmark = True


from sklearn.preprocessing import MinMaxScaler

""" Process data using a sliding window approach """
class GeneDataset(Dataset):
    def __init__(self, dataframe, use_sliding_window=False, window_size=100):
        self.dataframe = dataframe
        self.grouped_data = dataframe.groupby('ensembl_gene_id')
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size
        self.cache = {}
        self.windows = []

        # use subsequence windows from genes
        if self.use_sliding_window and window_size is not None:
            self._create_windows()
        # use full-length genes
        else:
            self._prepare_full_genes()
    
    def _create_windows(self):
        for gene_id, group in self.grouped_data:
            gene_length = len(group)
            for start_idx in range(0, gene_length - self.window_size + 1, self.window_size):
                end_idx = start_idx + self.window_size
                if end_idx > gene_length:
                    break
                window = group.iloc[start_idx:end_idx]
                self.windows.append((gene_id, window))
    
    def _prepare_full_genes(self):
        for gene_id, group in self.grouped_data:
            self.windows.append((gene_id, group))

    def __len__(self):
        return len(self.windows)

    # prepare single window or gene
    def __getitem__(self, idx):
        gene_id, window = self.windows[idx]
        
        if gene_id in self.cache:
            return self.cache[gene_id]
        
        strand_encoded = window['strand'].map({'-': 0, '+': 1}).values
        strand_tensor = torch.tensor(strand_encoded, dtype=torch.int64)
         
        result = {
            'GeneId': gene_id,
            'Chr': window['seqnames'].values,
            'Start': torch.tensor(window['start'].values, dtype=torch.int64),
            'End': torch.tensor(window['end'].values, dtype=torch.int64),
            'Strand': strand_tensor,
            
            # epigenomic features per gene j, site i
            'Y_ji':  torch.tensor(window[feature_names].values, dtype=torch.float64),
            
            # read counts per gene j, site i
            'X_ji': torch.tensor(window['score'].values, dtype=torch.float64),
            
            # read depth * initiation rate values per gene j
            'C_j': torch.tensor(window['lambda_alphaj'].iloc[0], dtype=torch.float64),
            
            # GLM elongation rate predictions per gene j, site i
            'Z_ji': torch.tensor(window['combined_zeta'].values, dtype=torch.float64),
            
            # one-hot encoded sequences
            'N_ji': torch.tensor(window[nucleotides].values, dtype=torch.float64), 
            'Length': len(window)
        }
    
        self.cache[gene_id] = result

        return result

""" Batch subsequences with same gene id together """
class GeneIdBatchSampler(BatchSampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.batches = self._create_batches()

    def _create_batches(self):
        # Group indices by GeneId
        gene_id_to_indices = {}
        for idx in range(len(self.dataset)):
            gene_id = self.dataset[idx]['GeneId']
            if gene_id not in gene_id_to_indices:
                gene_id_to_indices[gene_id] = []
            gene_id_to_indices[gene_id].append(idx)

        return list(gene_id_to_indices.values())

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

def custom_collate_fn(batch):
    gene_ids = [item['GeneId'] for item in batch]
    start = torch.stack([item['Start'] for item in batch])
    end = torch.stack([item['End'] for item in batch])
    strand = torch.stack([item['Strand'] for item in batch])
    Y_ji = torch.stack([item['Y_ji'] for item in batch])
    X_ji = torch.stack([item['X_ji'] for item in batch])
    C_j = torch.stack([item['C_j'] for item in batch])
    Z_ji = torch.stack([item['Z_ji'] for item in batch])
    N_ji = torch.stack([item['N_ji'] for item in batch])
    lengths = [item['Length'] for item in batch]
    
    # Handling lists of strings
    chrs = [item['Chr'] for item in batch]
    
    batched_data = {
        'GeneId': gene_ids,
        'Chr': chrs,
        'Start': start,
        'End': end,
        'Strand': strand,
        'Y_ji': Y_ji,
        'X_ji': X_ji,
        'C_j': C_j,
        'Z_ji': Z_ji,
        'N_ji': N_ji,
        'lengths': lengths
    }
    
    return batched_data


def build_dataset(data, batch_size, use_sliding_window, window_size=None):
    dataset = GeneDataset(data, use_sliding_window, window_size)
    #batch_sampler = GeneIdBatchSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=7, collate_fn=custom_collate_fn) #batch_sampler=batch_sampler, 
    return loader



def match_list_lengths(list1, list2):
    len1 = len(list1)
    len2 = len(list2)
    
    # If the first list is shorter, extend it
    if len1 < len2:
        list1.extend([list1[-1]] * (len2 - len1))
    # If the second list is shorter, extend it
    elif len2 < len1:
        list2.extend([list2[-1]] * (len1 - len2))
    
    return list1, list2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(config):

    class EpLinearModel(nn.Module):
        def __init__(self, input_size):
            super(EpLinearModel, self).__init__()
            self.name = "ep_linear"
            self.linear = nn.Linear(input_size, 1)

        def forward(self, Y_ji):
            x = self.linear(Y_ji)
            return x.squeeze(-1)   
    
    class EpSeqLinearModel(nn.Module):
        def __init__(self, num_ep_features, num_seq_features):
            super(EpSeqLinearModel, self).__init__()
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
        
    class CNN(nn.Module):
        def __init__(self, num_ep_features, num_seq_features, 
                     y_channels, y_kernel_sizes,
                     n_channels, n_kernel_sizes, dropout, 
                     lstm_layer_size, num_lstm_layers=None, bidirectional=False):
            
            super(CNN, self).__init__()
            self.name = "cnn"            

            self.y_convs = nn.ModuleList()
            y_in_channels = num_ep_features
            
            y_channels, y_kernel_sizes = match_list_lengths(y_channels, y_kernel_sizes)

            # Y_ji convolutional layers
            for idx, out_channels in enumerate(y_channels):
                self.y_convs.append(
                    nn.Conv1d(y_in_channels, out_channels, y_kernel_sizes[idx], stride=1, padding='same')
                )
                y_in_channels = out_channels
            
            
            self.n_convs = nn.ModuleList()
            n_in_channels = num_seq_features
            
            n_channels, n_kernel_sizes = match_list_lengths(n_channels, n_kernel_sizes)
            
            for idx, out_channels in enumerate(n_channels):
                self.n_convs.append(
                    nn.Conv1d(n_in_channels, out_channels, n_kernel_sizes[idx], stride=1, padding='same')
                )
                n_in_channels = out_channels

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            # Final convolutional layer to map to a single output channel
            # Since the output needs to be (batch_size, seq_len), we map the final features to 1
            self.final_conv = nn.Conv1d(y_channels[-1] + n_channels[-1], 1, 1)  # 1x1 convolution
            
            self.num_lstm_layers = num_lstm_layers
            if num_lstm_layers != 0 and num_lstm_layers != None:
                self.gru = nn.GRU(input_size=y_channels[-1] + n_channels[-1], hidden_size=lstm_layer_size, num_layers=num_lstm_layers, bidirectional=bidirectional, batch_first=True)
            
            self.final_linear = nn.Linear(lstm_layer_size, 1)
            self.bidirectional = True
            self.final_bidirectional_linear = nn.Linear(lstm_layer_size*2, 1)
            #self.batch_norm = nn.BatchNorm1d(y_hidden_layer_sizes[-1] + n_hidden_layer_sizes[-1])
            
        def forward(self, Y_ji, N_ji):
            Y_ji = Y_ji.permute(0, 2, 1)  
            N_ji = N_ji.permute(0, 2, 1)
            
            for conv in self.y_convs:
                Y_ji = conv(Y_ji)
                Y_ji = self.relu(Y_ji)
                Y_ji = self.dropout(Y_ji)
            
            for conv in self.n_convs:
                N_ji = conv(N_ji)
                N_ji = self.relu(N_ji)
                N_ji = self.dropout(N_ji)

            x = torch.cat((Y_ji, N_ji), 1)
            
            #x = self.batch_norm(x)
            
            if self.num_lstm_layers != 0 and self.num_lstm_layers != None:
                x = x.permute(0,2,1)
                x, (h_n, c_n) = self.gru(x)
                if self.bidirectional:
                    x = self.final_bidirectional_linear(x)
                else:
                    x = self.final_linear(x)
                x = x.squeeze(-1)
            
            else:
                x = self.final_conv(x)
                x = x.squeeze(1)  
                
            return x
    
    if config["model_type"] == 'ep_seq_linear':
        model = EpSeqLinearModel(num_features, num_seq_features)
    elif config["model_type"] == 'ep_linear':
        model = EpLinearModel(num_features)
    elif config["model_type"] == 'cnn':
        lstm_layer_size = None
        bidirectional = None
        if config["num_lstm_layers"] != 0 and config["num_lstm_layers"] != None:
            lstm_layer_size = config["lstm_layer_size"]
            bidirectional = config["bidirectional"]
        model = CNN(num_features, num_seq_features, config["y_channels"], config["y_kernel_sizes"], 
                    config["n_channels"], config["n_kernel_sizes"], config["dropout"], 
                    config["num_lstm_layers"], lstm_layer_size, bidirectional)
    
    if cuda_available:
        if num_gpus > 1:
            print("Using", num_gpus, "GPUs")
            model = torch.nn.DataParallel(model)
        model = model.to('cuda')

    print(model)

    
    model.double()

    return model.to(device)


def build_optimizer(network, learning_rate, l2_lambda):
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    return optimizer



class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, X_ji, C_j, rho_ji):
        loss = X_ji * rho_ji + C_j * torch.exp(-rho_ji) - X_ji * torch.log(C_j)
        
        # calculate average loss per site
        return (loss).mean()


print(config_file_name)
with open(config_folder_path + config_file_name + ".json", 'r') as file:
    config = json.load(file)


# load model state

model = build_model(config)

model.load_state_dict(torch.load(f"./models/{config_file_name}.pth", map_location=torch.device('cpu')))

cuda_available = torch.cuda.is_available()
print("CUDA (GPU support) is available:", cuda_available)
num_gpus = torch.cuda.device_count()
print("Number of GPUs available:", num_gpus)
if cuda_available:
    if num_gpus > 1:
        print("Using", num_gpus, "GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to('cuda')

first_param_device = next(model.parameters()).device
print("Model is on device:", first_param_device)

model.double()


""" Convert results to bigwig format """



#full_data = pd.concat([train_data, valid_data, test_data], ignore_index=True)



model.eval()

full_dl = build_dataset(test_data, 1, False, None) #full_data
full_data_iter = iter(full_dl)




"""
bw = pyBigWig.open(f"{cell_type}_neuralNetEpAllmerPredZeta.bw", "w")
chrom_sizes = {'chr1': 248956422, 'chr2': 242193529}
bw.addHeader(list(chrom_sizes.items()))
# Add entries from DataFrame
for index, row in df.iterrows():
    bw.addEntries([row['chrom']], [row['start']], ends=[row['end']], values=[row['value']])

# Close the BigWig file
bw.close()
"""


# In[ ]:


columns = ['Chr', 'Start', 'End', 'Value', 'Strand']
bw_df = df = pd.DataFrame(columns=columns)

bw_items = []
with torch.no_grad():
    for idx, batch in enumerate(full_dl):
        if idx % 100 == 0:
            print("calculating...")
        Y_ji = batch['Y_ji'].to(device)
        N_ji = batch['N_ji'].to(device)
        Z_ji = batch['Z_ji'].to(device)
        
        if model.name == "ep_linear":
            rho_ji = model(Y_ji)
        else:
            rho_ji = model(Y_ji, N_ji)    
        
        data_dict = {
            "Chr": batch["Chr"][0].tolist(), # stored as string
            "Start": batch["Start"][0].tolist(),
            "End": batch["End"][0].tolist(),
            "Value": torch.exp(rho_ji.squeeze().cpu()).tolist(),
            "Strand": batch["Strand"][0].tolist()
        }
        bw_items.append(pd.DataFrame(data_dict))

bw_df = pd.concat(bw_items, ignore_index=True)

print(bw_df.head())

# check that minus strand has only positive values
# Filter the DataFrame to include only rows where Strand is 0
minus_strand_df = bw_df[bw_df["Strand"] == 0]

# Check if all values in the Value column of this subset are greater than 0
all_values_positive = (minus_strand_df["Value"] > 0).all()

if not all_values_positive:
    print("Minus strand contains negative values before being processed")
# throw error if all_values_positive is false

# Remove minus strand when positive strand at same position
def filter_strands(group):
    # Mask for rows with Strand == 1
    mask_strand_1 = group['Strand'] == 1
    # Identify unique Start positions for Strand == 1 within the group
    starts_with_strand_1 = group.loc[mask_strand_1, 'Start'].unique()
    # Mask for rows with Strand == 0 and Start not in starts_with_strand_1, within the group
    mask_strand_0_unique_starts = (group['Strand'] == 0) & (~group['Start'].isin(starts_with_strand_1))
    # Combine masks to filter the group
    filtered_group = group[mask_strand_1 | mask_strand_0_unique_starts]
    return filtered_group

bw_df = bw_df.groupby('Chr', group_keys=False).apply(filter_strands).reset_index(drop=True)

# negate zeta value for minus strand
bw_df.loc[bw_df["Strand"] == 0, "Value"] = bw_df.loc[bw_df["Strand"] == 0, "Value"] * -1

# remove strand column when storing to bigwig file
del bw_df["Strand"]

# update Chr column values
bw_df['Chr'] = 'chr' + bw_df['Chr'].astype(str)

# setup header
epAllmer_bw = pyBigWig.open("./data/k562_epAllmerPredZeta.bw")
chrom_lengths = epAllmer_bw.chroms()
chrom_lengths = list(chrom_lengths.items())

bw = pyBigWig.open("./data/k562_epAllmerNeuralNetZeta.bw", "w")
bw.addHeader(chrom_lengths)

print("add entries to bw")

print(type(bw_df['Chr'].tolist()[0]))
print(type(bw_df['Start'].tolist()[0]))
print(type(bw_df['End'].tolist()[0]))
print(type(bw_df['Value'].tolist()[0]))

print(bw_df['Chr'].tolist()[:5])
print(bw_df['Start'].tolist()[:5])
print(bw_df['End'].tolist()[:5])
print(bw_df['Value'].tolist()[:5])

bw.addEntries(bw_df['Chr'].tolist(), bw_df['Start'].tolist(), ends=bw_df['End'].tolist(), values=bw_df['Value'].tolist())

print("finish adding entries to bw")

bw.close()
