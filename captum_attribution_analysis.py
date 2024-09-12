import torch
from data_processing.load_data import read_pickle, setup_dataloader
from load_model_checkpoint import load_model_checkpoint
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

nucleotides = ['A', 'T', 'G', 'C']
test_batch_size = 1

train_data, valid_data, test_data = read_pickle("k562_datasets")
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

test_dl = setup_dataloader(test_data, feature_names, nucleotides, test_batch_size, False, None)

# Initialize Integrated Gradients
integrated_gradients = IntegratedGradients(model)

epigenomic_attributions = []
sequence_attributions = []
for idx, batch in enumerate(test_dl):
    if idx == 50:
        break
    print(f"batch idx: {idx}")
    Y_ji = batch['Y_ji'].to(device)
    N_ji = batch['N_ji'].to(device)

    for target_index in range(len(batch)):
        attributions, delta = integrated_gradients.attribute((Y_ji, N_ji), target=target_index, return_convergence_delta=True)
        epigenomic_attributions.append(attributions[0].squeeze(0).cpu().detach().numpy())
        sequence_attributions.append(attributions[1].squeeze(0).cpu().detach().numpy())
        
epigenomic_attributions = np.concatenate(epigenomic_attributions, axis=0)
sequence_attributions = np.concatenate(sequence_attributions, axis=0)

combined_feature_names = list(feature_names) + nucleotides
data_to_save = {'epigenomic_attributions': epigenomic_attributions, 'sequence_attributions':sequence_attributions,'feature_names': combined_feature_names}
with open('captum_gradient_values_full.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)
    
# Average attributions across all predictions
avg_epigenomic_attributions = np.mean(epigenomic_attributions, axis=0)
avg_sequence_attributions = np.mean(sequence_attributions, axis=0)

data_to_save = {'avg_epigenomic_attributions': avg_epigenomic_attributions, 'avg_sequence_attributions': avg_sequence_attributions,'feature_names': combined_feature_names}
with open('captum_avg_gradient_values.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)
    
# Sum attributions across all predictions
sum_epigenomic_attributions = np.mean(epigenomic_attributions, axis=0)
sum_sequence_attributions = np.mean(sequence_attributions, axis=0)

data_to_save = {'sum_epigenomic_attributions': sum_epigenomic_attributions, 'sum_sequence_attributions': sum_sequence_attributions,'feature_names': combined_feature_names}
with open('captum_sum_gradient_values.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)

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
plt.savefig("attributions_testing/avg_attributions_performance_analysis_dataset_take2.png")
with open("attributions_testing/avg_attributions_performance_analysis_dataset_take2.txt", 'w') as f:
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


# Plotting the results
plt.figure(figsize=(14, 7))

# Epigenomic features
plt.subplot(1, 2, 1)
plt.bar(range(sum_epigenomic_attributions.shape[0]), sum_epigenomic_attributions, tick_label=feature_names)
plt.xticks(rotation=45, ha='right')
plt.title('Summed Epigenomic Feature Attributions')
plt.xlabel('Epigenomic Feature Index')
plt.ylabel('Summed Attributions')

# Sequence features
plt.subplot(1, 2, 2)
plt.bar(range(sum_sequence_attributions.shape[0]), sum_sequence_attributions, tick_label=nucleotides)
plt.title('Summed Sequence Feature Attributions')
plt.xlabel('Sequence Feature Index')
plt.ylabel('Summed Attribution')

plt.tight_layout()
plt.savefig("attributions_testing/sum_attributions_performance_analysis_dataset_take2.png")
with open("attributions_testing/sum_attributions_performance_analysis_dataset_take2.txt", 'w') as f:
    # Write the text
    f.write("Epigenomic Feature Attributions: \n")
    for idx, feature in enumerate(feature_names):
        f.write(f"{feature}: {sum_epigenomic_attributions[idx]:.2e}")
        f.write("\n")
    f.write("\n\n")
    f.write("Sequence Feature Attributions: \n")
    for idx, nucleotide in enumerate(nucleotides):
        f.write(f"{nucleotide}: {sum_sequence_attributions[idx]:.2e}")
        f.write("\n")
    f.write("\n\n")
plt.show()