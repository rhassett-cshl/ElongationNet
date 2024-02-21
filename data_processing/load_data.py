import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import pickle
from .GeneDataset import GeneDataset
from .BucketBatchSampler import BucketBatchSampler
from .BucketGeneDataset import BucketGeneDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def read_pickle(cell_type):
    with open(f'./data/{cell_type}_datasets.pkl', 'rb') as file:
        combined_datasets = pickle.load(file)
        
    train_data = combined_datasets['train']
    valid_data = combined_datasets['valid']
    test_data = combined_datasets['test']

    print(train_data.iloc[0])
    
    return train_data, valid_data, test_data

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

def padded_collate_fn(batch):
    GeneIds, Seq_Names, Start, End, Strand, C_j, Lengths = zip(*[(item['GeneId'], item['Seq_Name'], item['Start'], item['End'], item['Strand'], item['C_j'], item['Length']) for item in batch])
    
    Y_ji = pad_sequence([item['Y_ji'] for item in batch], batch_first=True, padding_value=0.0)
    X_ji = pad_sequence([item['X_ji'] for item in batch], batch_first=True, padding_value=0.0)
    Z_ji = pad_sequence([item['Z_ji'] for item in batch], batch_first=True, padding_value=1.0)
    N_ji = pad_sequence([item['N_ji'] for item in batch], batch_first=True, padding_value=-1)
    
    longest_seq_length = torch.arange(X_ji.size(1)).unsqueeze(0)
    seq_lengths = torch.tensor(Lengths).unsqueeze(-1) 
    mask = longest_seq_length < seq_lengths
    
    return {
        'GeneId': GeneIds,
        'Seq_Name': Seq_Names,
        'Start': Start,
        'End': End,
        'Strand': Strand,
        'Y_ji': Y_ji,
        'X_ji': X_ji,
        'C_j': torch.stack(C_j).unsqueeze(1),
        'Z_ji': Z_ji,
        'N_ji': N_ji,
        'Mask': mask,
        'Length': len(X_ji[0])
    }

def setup_dataloader(data, feature_names, nucleotides, 
                     batch_size, use_sliding_window, window_size=100):
    
    #dataset = GeneDataset(data, feature_names, nucleotides, use_sliding_window, window_size)
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=7, collate_fn=custom_collate_fn)
    
    dataset = BucketGeneDataset(data, feature_names, nucleotides)
    batch_sampler = BucketBatchSampler(dataset, 128, 25)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=7, collate_fn=padded_collate_fn)
    
    return loader
