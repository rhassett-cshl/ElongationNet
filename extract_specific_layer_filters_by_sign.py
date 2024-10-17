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

nucleotides = ['A', 'T', 'G', 'C']
nucleotide_map = {nucleotide: idx for idx, nucleotide in enumerate(nucleotides)}

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

conv_layer_id = 0

seq_conv_layer = model.n_convs[conv_layer_id]

model_weights = seq_conv_layer.weight.data.numpy()

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_conv_output_for_seq(y, seq, conv_layer):
    '''
    Given an input sequeunce and a convolutional layer, 
    get the output tensor containing the conv filter 
    activations along each position in the sequence
    '''
    
    hook = model.n_convs[conv_layer_id].register_forward_hook(get_activation(f"n_convs_{conv_layer_id}"))

    # run seq through conv layer
    with torch.no_grad(): # don't want as part of gradient graph
        # apply learned filters to input seq
        output = model(y.unsqueeze(0),seq.unsqueeze(0))
        specific_layer_activations = activation[f"n_convs_{conv_layer_id}"]
        hook.remove()
        return specific_layer_activations[0]
    

def get_filter_activations(Y_ji, seqs, conv_layer,act_thresh=0):
    '''
    Given a set of input sequences and a trained convolutional layer, 
    determine the subsequences for which each filter in the conv layer 
    activate most strongly. 
    
    1.) Run seq inputs through conv layer. 
    2.) Loop through filter activations of the resulting tensor, saving the
            position where filter activations were > act_thresh. 
    3.) Compile a count matrix for each filter by accumulating subsequences which
            activate the filter above the threshold act_thresh
    '''
    # initialize dict of pwms for each filter in the conv layer
    # pwm shape: 4 nucleotides X filter width, initialize to 0.0s
    num_filters = conv_layer.out_channels
    filt_width = conv_layer.kernel_size[0]
    filter_pwms = dict((i,torch.zeros(4,filt_width)) for i in range(num_filters))
    filter_pwms_neg = dict((i,torch.zeros(4,filt_width)) for i in range(num_filters))
    
    print("Num filters", num_filters)
    print("filt_width", filt_width)
    
    # loop through a set of sequences and collect subseqs where each filter activated
    for y, seq in zip(Y_ji, seqs):
        # get a tensor of each conv filter activation along the input seq
        res = get_conv_output_for_seq(y, seq, conv_layer)

        # for each filter and it's activation vector
        for filt_id,act_vec in enumerate(res):
            for pos in range(0,len(seq)-filt_width):
                subseq = seq[pos:pos+filt_width]
                filt_act = act_vec[pos:pos+filt_width]
                # transpose OHE to match PWM orientation
                subseq_tensor = subseq.permute(1,0)

                # add this subseq to the pwm count for this filter
                # check that all abs values in motif are above threshold
                if subseq_tensor.shape == (4, filt_width) and torch.all(torch.abs(filt_act) > act_thresh):
                    filter_pwms[filt_id] += subseq_tensor * filt_act 
    
    # standardize filter pwms  
    all_activations = torch.cat([filter_pwms[filt_id] for filt_id in filter_pwms])
    mean = all_activations.mean()
    std = all_activations.std()
    
    for filt_id in filter_pwms:
        filter_pwms[filt_id] = (filter_pwms[filt_id] - mean) / std
         
    return filter_pwms

def view_filters_and_logos(model_weights,filter_activations, num_cols=8):
    '''
    Given some convolutional model weights and filter activation PWMs, 
    visualize the heatmap and motif logo pairs in a simple grid
    '''
    print(model_weights.shape)

    # make sure the model weights agree with the number of filters
    assert(model_weights.shape[0] == len(filter_activations))
    
    num_filts = len(filter_activations)
    num_rows = int(np.ceil(num_filts/num_cols))*2+1 
    # ^ not sure why +1 is needed... complained otherwise
    
    plt.figure(figsize=(20, 17))

    j=0 # use to make sure a filter and it's logo end up vertically paired
    for i, filter in enumerate(model_weights):
        if (i)%num_cols == 0:
            j += num_cols

        # display sequence logo
        ax2 = plt.subplot(num_rows, num_cols, i+j+1+num_cols)
        filt_df = pd.DataFrame(filter_activations[i].T.numpy(),columns=nucleotides)
        logo = logomaker.Logo(filt_df,ax=ax2)
        ax2.set_title(f"Filter {i}")

    plt.tight_layout()
    plt.savefig("first_layer_filters_sign6.png")

filter_activations = get_filter_activations(Y_ji, N_ji, seq_conv_layer,act_thresh=0.9)
view_filters_and_logos(model_weights,filter_activations)