import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

""" Process data using a sliding window approach """
class GeneDataset(Dataset):
    def __init__(self, dataframe, feature_names, nucleotides, 
                 use_sliding_window=False, window_size=100):
        self.dataframe = dataframe
        self.grouped_data = dataframe.groupby('ensembl_gene_id')
        self.feature_names = feature_names
        self.nucleotides = nucleotides
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
            'Y_ji':  torch.tensor(window[self.feature_names].values, dtype=torch.float64),
            
            # read counts per gene j, site i
            'X_ji': torch.tensor(window['score'].values, dtype=torch.float64),
            
            # read depth * initiation rate values per gene j
            'C_j': torch.tensor(window['combined_lambda_alphaj'].iloc[0], dtype=torch.float64),
            
            # GLM elongation rate predictions per gene j, site i
            'Z_ji': torch.tensor(window['combined_zeta'].values, dtype=torch.float64),
            
            # one-hot encoded sequences
            'N_ji': torch.tensor(window[self.nucleotides].values, dtype=torch.float64)
        }
    
        self.cache[gene_id] = result

        return result
