import torch
from data_processing.load_data import read_pickle, setup_dataloader
from load_model_checkpoint import load_model_checkpoint
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle

train_data, valid_data, test_data = read_pickle("k562_performance_analysis_datasets")
nucleotides = ['A', 'T', 'G', 'C']
test_batch_size = 500
train_batch_size = 500
column_names = np.array(train_data.columns)
feature_names = column_names[6:16]
test_dl = setup_dataloader(test_data, feature_names, nucleotides, test_batch_size, True, 100)

first_test_batch = next(iter(test_dl))
test_N_ji = first_test_batch['N_ji']

with open('/home/hassett/ElongationNet/shap_gradient_values_batch1.pkl', 'rb') as file:
    data = pickle.load(file)

print("Finish loading shap values")

# Take mean or sum along feature dimension 3

seq_shap_values_aggregated = np.sum(data["shap_values"][1], axis=3) #np.mean
seq_shap_values_aggregated = np.transpose(seq_shap_values_aggregated, (0, 2, 1))
np.savez('shap_attribution_values_summed', seq_shap_values_aggregated)