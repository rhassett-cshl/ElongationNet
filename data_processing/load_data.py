import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from GeneDataset import GeneDataset
from GeneBatchSampler import GeneBatchSampler
from torch.utils.data import DataLoader

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
        'N_ji': N_ji
    }
    
    return batched_data

def load_data(data, batch_size, use_sliding_window, window_size=None):
    dataset = GeneDataset(data, use_sliding_window, window_size)
    #batch_sampler = GeneBatchSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=7, collate_fn=custom_collate_fn) #batch_sampler=batch_sampler,
    return loader
