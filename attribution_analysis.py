import torch
from data_processing.load_data import read_pickle, setup_dataloader
from load_model_checkpoint import load_model_checkpoint
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import numpy as np
import matplotlib.pyplot as plt
import json

nucleotides = ['A', 'T', 'G', 'C']
test_batch_size = 1

train_data, valid_data, test_data = read_pickle("k562")
column_names = np.array(train_data.columns)
feature_names = column_names[6:16]
num_ep_features = len(feature_names)
num_seq_features = len(nucleotides)

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

with open("./configs/elongation_net_v1.json", 'r') as file:
    config = json.load(file)

model = load_model_checkpoint("elongation_net_v1", config, device, num_ep_features, num_seq_features)

model.eval()

test_dl = setup_dataloader(test_data, feature_names, nucleotides, test_batch_size, False, None)

# Initialize Integrated Gradients
integrated_gradients = IntegratedGradients(model)

first_batch = next(iter(test_dl))
Y_ji = first_batch['Y_ji'].to(device)
N_ji = first_batch['N_ji'].to(device)

# Compute attributions
epigenomic_attributions = []
sequence_attributions = []
for target_index in range(len(first_batch)):
    attributions, delta = integrated_gradients.attribute((Y_ji, N_ji), target=target_index, return_convergence_delta=True)
    epigenomic_attributions.append(attributions[0])
    sequence_attributions.append(attributions[1])
    print(attributions)
    #all_deltas.append(delta)

# at some point will not want to remove the batch dimension
epigenomic_attributions = np.array(epigenomic_attributions[0].squeeze(0))
sequence_attributions = np.array(sequence_attributions[0].squeeze(0))

# Average attributions across all predictions
avg_epigenomic_attributions = np.mean(epigenomic_attributions, axis=0)
avg_sequence_attributions = np.mean(sequence_attributions, axis=0)

# Plotting the results
plt.figure(figsize=(14, 7))

# Epigenomic features
plt.subplot(1, 2, 1)
plt.bar(range(avg_epigenomic_attributions.shape[0]), avg_epigenomic_attributions, tick_label=feature_names)
plt.xticks(rotation=45, ha='right')
plt.title('Average Epigenomic Feature Attributions')
plt.xlabel('Epigenomic Feature Index')
plt.ylabel('Average Attribution')

# Sequence features
plt.subplot(1, 2, 2)
plt.bar(range(avg_sequence_attributions.shape[0]), avg_sequence_attributions, tick_label=nucleotides)
plt.title('Average Sequence Feature Attributions')
plt.xlabel('Sequence Feature Index')
plt.ylabel('Average Attribution')

plt.tight_layout()
plt.savefig("attributions_testing/batchsize_1.png")
with open("attributions_testing/batchsize_1.txt", 'w') as f:
    # Write the text
    f.write("Epigenomic Feature Attributions: \n")
    for idx, feature in enumerate(feature_names):
        f.write(f"{feature}: {avg_epigenomic_attributions[idx]:.2e}")
        f.write("\n")
    f.write("\n\n")
    f.write("Sequence Feature Attributions: \n")
    for idx, nucleotide in enumerate(nucleotides):
        f.write(f"{nucleotide}: {avg_sequence_attributions[idx]:.2e}")
        f.write("\n")
    f.write("\n\n")
plt.show()