import numpy as np
import pandas as pd
import torch
import os
from bigwig_utilities.process_bigwig_dataframe import process_bigwig_dataframe
from bigwig_utilities.save_results_to_bigwig import save_to_bigwig
from load_model_checkpoint import load_model_checkpoint
from data_processing.load_data import read_pickle, setup_dataloader

nucleotides = ['A', 'T', 'G', 'C']
test_batch_size = 1
bw_columns = ['Chr', 'Start', 'End', 'Value', 'Strand']

def save_results(config_name, config):
    train_data, valid_data, test_data = read_pickle(config["cell_type"])

    column_names = np.array(train_data.columns)
    feature_names = column_names[6:16]
    num_ep_features = len(feature_names)
    num_seq_features = len(nucleotides)
    
    cuda_available = torch.cuda.is_available()
    
    device = torch.device("cuda" if cuda_available else "cpu")
    
    model = load_model_checkpoint(config_name, config, device, num_ep_features, num_seq_features)
    
    os.makedirs(f"./results/{config_name}", exist_ok=True)
 
    model.eval()
       
    full_data = pd.concat([train_data, valid_data, test_data], ignore_index=True)
    full_dl = setup_dataloader(full_data, feature_names, nucleotides, test_batch_size, False, None)

    use_ep_linear = model.name == "ep_linear"

    bw_columns = ["Chr", "Start", "End", "Value", "Strand"]
    bw_data = {col: [] for col in bw_columns}
    with torch.no_grad():
        for batch in full_dl:
            Y_ji = batch['Y_ji'].to(device)
            N_ji = batch['N_ji'].to(device)
            Z_ji = batch['Z_ji'].to(device)
            
            if use_ep_linear:
                rho_ji = model(Y_ji)
            else:
                rho_ji = model(Y_ji, N_ji)    

            bw_data["Chr"].extend(batch["Chr"][0])
            bw_data["Start"].extend(batch["Start"][0].numpy())
            bw_data["End"].extend(batch["End"][0].numpy())
            bw_data["Value"].extend(torch.exp(rho_ji.squeeze().cpu()).numpy())
            bw_data["Strand"].extend(batch["Strand"][0].numpy())


    
    bw_df = pd.DataFrame(bw_data)


    #bw_df = pd.concat(bw_items, ignore_index=True)
   
    print("concat results finished")
 
    bw_df = process_bigwig_dataframe(bw_df)
    
    save_to_bigwig(bw_df, config_name, config["cell_type"])
