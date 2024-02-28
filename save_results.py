import numpy as np
import pandas as pd
import torch
import os
from bigwig_utilities.process_bigwig_dataframe import process_bigwig_dataframe
from bigwig_utilities.save_results_to_bigwig import save_to_bigwig
from load_model_checkpoint import load_model_checkpoint
from data_processing.load_data import read_pickle, setup_dataloader
from loss_metrics import CustomLoss

nucleotides = ['A', 'T', 'G', 'C']
test_batch_size = 1
bw_columns = ['Chr', 'Start', 'End', 'Value', 'Strand']

save_bigwig = False

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
    full_dl = setup_dataloader(test_data, feature_names, nucleotides, test_batch_size, False, None)

    use_ep_linear = model.name == "ep_linear"
    
    loss_fn = CustomLoss()

    bw_columns = ["Chr", "Start", "End", "Value", "Strand"]
    csv_columns = ["Chr", "Start", "End", "Strand", "GeneId", "X_ji", "C_j", "Predicted_Zeta", "GLM_Combined_Zeta", "GLM_Epft_Zeta", "Loss"]
    bw_data = {col: [] for col in bw_columns}
    csv_data = {col: [] for col in csv_columns}
    with torch.no_grad():
        for idx, batch in enumerate(full_dl):
            if idx % 100 == 0:
                print("calculating..")
            Y_ji = batch['Y_ji'].to(device)
            X_ji = batch['X_ji'].to(device)
            C_j = batch['C_j'].to(device)
            N_ji = batch['N_ji'].to(device)
            
            if use_ep_linear:
                rho_ji = model(Y_ji)
            else:
                rho_ji = model(Y_ji, N_ji) 
                
            avg_loss, loss_items = loss_fn(X_ji, C_j, rho_ji) 
            print(f"CNN Avg Test Loss: {avg_loss}\n")  
            avg_loss2, loss_items = loss_fn(X_ji, C_j, torch.log(batch["Z_ji"][0])) 
            print(f"GLM Avg Test Loss: {avg_loss2}\n")  

            if save_bigwig:
                bw_data["Chr"].extend(batch["Chr"][0])
                bw_data["Start"].extend(batch["Start"][0].numpy())
                bw_data["End"].extend(batch["End"][0].numpy())
                bw_data["Value"].extend(torch.exp(rho_ji.squeeze().cpu()).numpy())
                bw_data["Strand"].extend(batch["Strand"][0].numpy())
            
            batch_size = len(batch["Start"][0])

            csv_data["Chr"].extend(batch["Chr"][0])
            csv_data["Start"].extend(batch["Start"][0].numpy())
            csv_data["End"].extend(batch["End"][0].numpy())
            csv_data["Predicted_Zeta"].extend(torch.exp(rho_ji.squeeze().cpu()).numpy())
            csv_data["GLM_Combined_Zeta"].extend(batch["Z_ji"][0].numpy())
            csv_data["GLM_Epft_Zeta"].extend(batch["epft_Z_ji"][0].numpy())
            csv_data["Strand"].extend(batch["Strand"][0].numpy())
            csv_data["X_ji"].extend(batch["X_ji"][0].numpy())
            csv_data["Loss"].extend(loss_items[0].numpy())

            csv_data["C_j"].extend([batch["C_j"][0].numpy()] * batch_size)
            csv_data["GeneId"].extend([batch["GeneId"][0]] * batch_size)

            
    csv_df = pd.DataFrame(csv_data)

    csv_df.to_csv(f'./results/{config_name}/{config_name}_results.csv', index=False)

    print("converted to csv file")

    if save_bigwig:
        bw_df = pd.DataFrame(bw_data)
    
        bw_df = process_bigwig_dataframe(bw_df)
    
        save_to_bigwig(bw_df, config_name, config["cell_type"])
