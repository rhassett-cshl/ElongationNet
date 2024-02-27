import numpy as np
import pandas as pd
import pickle
import torch
import os
import json
from bigwig_utilities.process_bigwig_dataframe import process_bigwig_dataframe
from bigwig_utilities.save_results_to_bigwig import save_to_bigwig
from load_model_checkpoint import load_model_checkpoint
from data_processing.load_data import read_pickle, setup_dataloader
from loss_debug import CustomLoss

config_name = "ep_seq_linear_1"  #"cnn_2" # ep_seq_linear_1

with open("./configs/" + config_name + ".json", 'r') as file:
    config = json.load(file)

nucleotides = ['A', 'T', 'G', 'C']
test_batch_size = 1

with open(f'./data/k562_datasets.pkl', 'rb') as file:
    combined_datasets = pickle.load(file)
    
test_data = combined_datasets['test']

column_names = np.array(test_data.columns)
feature_names = column_names[6:16]
num_ep_features = len(feature_names)
num_seq_features = len(nucleotides)

cuda_available = torch.cuda.is_available()

device = torch.device("cuda" if cuda_available else "cpu")

model = load_model_checkpoint(config_name, config, device, num_ep_features, num_seq_features)

model.eval()
    
test_dl = setup_dataloader(test_data, feature_names, nucleotides, test_batch_size, False, None)

loss_fn = CustomLoss()

csv_columns = ["Chr", "Start", "Strand", "GeneId", "Expected_X_ji", "C_j", "CNN_Zeta", "Loss", "GLM_Combined_Zeta"]#, "Predicted_X_ji"]
csv_data = {col: [] for col in csv_columns}
with torch.no_grad():
    for idx, batch in enumerate(test_dl):
        if idx % 100 == 0:
            print("calculating..")
        Y_ji = batch['Y_ji'].to(device)
        X_ji = batch['X_ji'].to(device)
        C_j = batch['C_j'].to(device)
        N_ji = batch['N_ji'].to(device)
        Z_ji = batch['Z_ji'].to(device)
        
        rho_ji = model(Y_ji, N_ji)    
        
        avg_loss, loss_items = loss_fn(X_ji, C_j, rho_ji)
        
        batch_size = len(batch["Start"][0])

        csv_data["Chr"].extend(batch["Chr"][0])
        csv_data["Start"].extend(batch["Start"][0].numpy())
        csv_data["Rho_ji"].extend(rho_ji.squeeze().cpu().numpy())
        csv_data["CNN_Zeta"].extend(torch.exp(rho_ji.squeeze().cpu()).numpy())
        csv_data["GLM_Combined_Zeta"].extend(batch["Z_ji"][0].numpy())
        csv_data["Strand"].extend(batch["Strand"][0].numpy())
        csv_data["Loss"].extend(loss_items[0].numpy())
        csv_data["Expected_X_ji"].extend(batch["X_ji"][0].numpy())

        csv_data["C_j"].extend([batch["C_j"][0].numpy()] * batch_size)
        csv_data["GeneId"].extend([batch["GeneId"][0]] * batch_size)

        #predicted_xji = ([batch["C_j"][0].numpy()] * batch_size) / batch["Z_ji"][0].numpy()

        #csv_data["Predicted_X_ji"].extend(predicted_xji)

        

csv_df = pd.DataFrame(csv_data)

csv_df.to_csv(f'./results/{config_name}/predictions.csv', index=False)

print("converted to csv file")
