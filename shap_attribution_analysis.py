import torch
from data_processing.load_data import read_pickle, setup_dataloader
from load_model_checkpoint import load_model_checkpoint
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import numpy as np
import matplotlib.pyplot as plt
import json
import shap

nucleotides = ['A', 'T', 'G', 'C']
test_batch_size = 1
train_batch_size = 500

train_data, valid_data, test_data = read_pickle("k562_performance_analysis_datasets")
column_names = np.array(train_data.columns)
feature_names = column_names[6:16]
num_ep_features = len(feature_names)
num_seq_features = len(nucleotides)

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

with open("./configs/elongation_net_v1_performance_analysis.json", 'r') as file:
    config = json.load(file)

model = load_model_checkpoint("elongation_net_v1_performance_analysis", config, device, num_ep_features, num_seq_features)

model.eval()
train_window_size = None
if config["train_use_sliding_window"]:
    train_window_size = config["train_window_size"]
train_dl = setup_dataloader(train_data, feature_names, nucleotides, train_batch_size, False, None)
test_dl = setup_dataloader(test_data, feature_names, nucleotides, test_batch_size, False, None)

first_train_batch = next(iter(train_dl))
background_Y_ji = first_train_batch['Y_ji'].to(device)
background_N_ji = first_train_batch['N_ji'].to(device)
explainer = shap.DeepExplainer(model, (background_Y_ji, background_N_ji))

first_test_batch = next(iter(test_dl))
test_Y_ji = first_test_batch['Y_ji'].to(device)
test_N_ji = first_test_batch['N_ji'].to(device)
shap_values = explainer.shap_values((test_Y_ji, test_N_ji))