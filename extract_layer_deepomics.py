import torch
from data_processing.load_data import read_pickle, setup_dataloader
from load_model_checkpoint import load_model_checkpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import torch.nn.functional as F
import logomaker
from visualize import activation_pwm, plot_filter_logos

nucleotides = ['A', 'T', 'G', 'C']

device = "cpu"

test_batch_size=1
train_data, valid_data, test_data = read_pickle("k562_datasets")
column_names = np.array(train_data.columns)
feature_names = column_names[6:16]
num_ep_features = 10
num_seq_features = len(nucleotides)

with open("./configs/elongation_net_v1_performance_analysis.json", 'r') as file:
    config = json.load(file)

model = load_model_checkpoint("elongation_net_v1_performance_analysis", config, device, num_ep_features, num_seq_features)

model.eval()

train_data, valid_data, test_data = read_pickle("k562_datasets")

test_dl = setup_dataloader(valid_data, feature_names, nucleotides, test_batch_size, False, None)

first_batch = next(iter(test_dl))

N_ji = first_batch["N_ji"]
Y_ji = first_batch["Y_ji"]

conv_layer_id = 1
window_size = 5

seq_conv_layer = model.n_convs[conv_layer_id]

model_weights = seq_conv_layer.weight.data.numpy()

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_conv_output_for_seq(y, seq):
    hook = model.n_convs[conv_layer_id].register_forward_hook(get_activation(f"n_convs_{conv_layer_id}"))

    # run seq through conv layer
    with torch.no_grad(): # don't want as part of gradient graph
        # apply learned filters to input seq
        output = model(y,seq)
        specific_layer_activations = activation[f"n_convs_{conv_layer_id}"]
        hook.remove()
        return specific_layer_activations

#N,seq_length,_,num_dims = X.shape
X = N_ji.unsqueeze(0).permute(0, 2, 1, 3)

activations = get_conv_output_for_seq(Y_ji, N_ji)
#fmap = X.permute(3,0,2,1) * activations.unsqueeze(2)
fmap = activations.unsqueeze(2)
fmap = fmap.permute(0, 3, 2, 1)

W = activation_pwm(fmap.numpy(), X=X.numpy(), threshold=0.9, window=window_size)
print(W)
# plot with logomaker
# logomaker 2019, deepomics 2019

W = np.squeeze(W, axis=1)

def plot_logo_for_filter(seq_data, filter_idx):
    # Extract the slice for the chosen filter, shape will be (9, 4) (positions x nucleotides)
    filter_data = seq_data[:, :, filter_idx]

    # Convert to pandas DataFrame for logomaker
    # Columns: ['A', 'C', 'G', 'T'], Rows: corresponding positions (0 to 8 for filter size 9)
    df = pd.DataFrame(filter_data, columns=nucleotides)

    # Normalize the data (optional, depending on whether your data is already probabilities or scores)
    #df = df.div(df.sum(axis=1), axis=0)

    # Plot using logomaker
    plt.figure(figsize=(10, 4))
    logo = logomaker.Logo(df)
    plt.title(f'Sequence Logo for Filter {filter_idx}')
    plt.savefig(f"deepomics LAYER {conv_layer_id + 1} filter {filter_idx}")

for filter_idx in range(fmap.shape[-1]):
    plot_logo_for_filter(W, filter_idx)
