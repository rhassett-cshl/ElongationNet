import numpy as np
import pandas as pd
import torch

def valid_epoch(model, loader, loss_fn):
    model.eval()
    total_neural_net_loss = 0
    total_glm_loss = 0
    neural_net_zeta = []
    glm_zeta = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            Y_ji_batch = batch['Y_ji'].to(device)
            X_ji_batch = batch['X_ji'].to(device)
            N_ji_batch = batch['N_ji'].to(device) 
            C_j_batch = batch['C_j'].to(device).unsqueeze(1)
            Z_ji_batch = batch['Z_ji'].to(device)
            
            if model.name == "ep_linear":
                outputs = model(Y_ji_batch)
            else:
                outputs = model(Y_ji_batch, N_ji_batch)
            	
	    print("neural net")
            neural_net_loss = loss_fn(X_ji_batch, C_j_batch, outputs)
            print("glm")
            glm_loss = loss_fn(X_ji_batch, C_j_batch, torch.log(Z_ji_batch))

            total_neural_net_loss +=  neural_net_loss.item()
            total_glm_loss += glm_loss.item()
            
            # store all predictions in list
            neural_net_zeta.append(torch.exp(outputs.cpu().flatten()))
            glm_zeta.append(batch['Z_ji'].flatten())
    
    # calculate average loss across all batches
    avg_neural_net_loss = total_neural_net_loss / len(loader)
    avg_glm_loss = total_glm_loss / len(loader)
    
    neural_net_zeta = torch.cat(neural_net_zeta)
    glm_zeta = torch.cat(glm_zeta)
    
    return avg_neural_net_loss, avg_glm_loss, neural_net_zeta, glm_zeta
