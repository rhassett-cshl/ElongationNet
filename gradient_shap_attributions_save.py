import torch
from data_processing.load_data import read_pickle, setup_dataloader
from load_model_checkpoint import load_model_checkpoint
from captum.attr import GradientShap
from captum.attr import visualization as viz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
import logomaker

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

test_df = test_data[(test_data['seqnames'] == 1)]

test_dl = setup_dataloader(test_df, feature_names, nucleotides, test_batch_size, True, 100)

gradient_shap = GradientShap(model)

epi_attributions_list = []
seq_attributions_list = []

print(f"# batches: {len(test_dl)}")

for idx, batch in enumerate(test_dl):
    if idx == 1000:
        break
    print(f"Batch #: {idx}", flush=True)
    
    Y_ji = batch['Y_ji'].to(device)
    N_ji = batch['N_ji'].to(device)
    baseline_Y = torch.zeros_like(Y_ji) 
    baseline_N = torch.zeros_like(N_ji) 
    for target_idx in range(0, 100):
        attributions = gradient_shap.attribute(
            inputs=(Y_ji,N_ji),                    # The input you want to explain
            baselines=(baseline_Y, baseline_N),    # Baseline to compare input with
            target=target_idx,                     # Target output index for which attributions are computed
        )
        epi_attributions_list.append(attributions[0])
        #five_mers = attributions[1][:, (target_idx-2):(target_idx+3), :]
        seq_attributions_list.append(attributions[1])#torch.squeeze(five_mers, dim=0))
        

#np.savez('gradient_shap_epi_attributions_chr14_100batches.npz', attributions=np.array(epi_attributions_list))
np.savez('gradient_shap_seq_attributions_test_chr1_1000batches.npz', attributions=np.array(seq_attributions_list))

exit()

#for idx, sequence_attributions in enumerate(seq_attributions_list):
seq_attr = np.array(seq_attributions_list)
mean_importance = np.mean(seq_attr, axis=0)
df_sequence_attributions = pd.DataFrame(mean_importance, columns=nucleotides, index=[-2, -1, 0, 1, 2])

# Create a logo plot for the attributions
logomaker.Logo(df_sequence_attributions, shade_below=0.5, fade_below=0.5)

# Add title and show plot
plt.title(f'Averaged Sequence Attributions')
plt.xlabel('Position in Sequence')
plt.ylabel('Attribution Score')
plt.savefig(f'aggregated_attributions/5mer_mean_attributions_fulltest.png', bbox_inches='tight')

# Clear the figure to avoid overlap in the next iteration
plt.clf()