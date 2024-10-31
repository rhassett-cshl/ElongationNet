import torch
from data_processing.load_data import read_pickle, setup_dataloader
from load_model_checkpoint import load_model_checkpoint
from captum.attr import GradientShap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import logomaker
import os

nucleotide_to_index = {
    'A': 0,
    'T': 1,
    'G': 2,
    'C': 3
}

nucleotides = ['A', 'T', 'G', 'C']
test_batch_size = 1
train_data, valid_data, test_data = read_pickle("k562_datasets")
column_names = np.array(train_data.columns)
feature_names = column_names[6:16]
num_ep_features = len(feature_names)
num_seq_features = len(nucleotides)
device = "cpu"
with open("./configs/elongation_net_v1_performance_analysis.json", 'r') as file:
    config = json.load(file)
model = load_model_checkpoint("elongation_net_v1_performance_analysis", config, device, num_ep_features, num_seq_features)
model.eval()

# kmer mapped to target pos of active site
kmer_dict = {"CA": 0}
flank_len = 5
dir_name = "kmer_CA_attr_results_methylation"
os.makedirs(dir_name, exist_ok=True)

test_dl = setup_dataloader(test_data, feature_names, nucleotides, test_batch_size, False, None)

gradient_shap = GradientShap(model)

for motif, active_site_idx in kmer_dict.items():
    indices = [nucleotide_to_index[n] for n in motif]
    one_hot_motif = torch.eye(4)[indices]
    motif_length = one_hot_motif.size(0)
    motif_matches = []
    for idx, batch in enumerate(test_dl):

        Y_ji = batch['Y_ji'].squeeze(0)
        N_ji = batch['N_ji'].squeeze(0)
        seq_length = N_ji.size(0)
        
        for i in range(seq_length - motif_length + 1):
            window = N_ji[i:i+motif_length]
            if torch.all(window == one_hot_motif):
                center = i + active_site_idx
                start = max(0, center - flank_len)
                end = min(seq_length, center + flank_len + 1)   
                
                subsequence = N_ji[start:end]
                epigenomic_values = Y_ji[start:end]
                
                if len(subsequence) == (flank_len * 2) + 1:
                    motif_matches.append((epigenomic_values, subsequence))    
                
        print(f"Found {len(motif_matches)} matches.")     
        
    epi_attrs = []
    for match_idx, (Y_ji, N_ji) in enumerate(motif_matches):
        input = (Y_ji.unsqueeze(0), N_ji.unsqueeze(0))
        baseline_Y = torch.zeros_like(Y_ji).unsqueeze(0)
        baseline_N = torch.zeros_like(N_ji).unsqueeze(0)
        attributions = gradient_shap.attribute(
            inputs=input,                  
            baselines=(baseline_Y, baseline_N),    
            target=flank_len,                     
        )
        
        epi_attr = attributions[0][15].squeeze().detach().numpy()
        epi_attrs.append(epi_attr)
        
    attr_avg = np.mean(epi_attrs, axis=0)
