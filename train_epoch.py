import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def train_epoch(model, loader, device, optimizer, loss_fn, l1_lambda):
    model.train()
    total_loss = 0
    for idx, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        Y_ji_batch = batch['Y_ji'].to(device) 
        X_ji_batch = batch['X_ji'].to(device)
        N_ji_batch = batch['N_ji'].to(device) 
        C_j_batch = batch['C_j'].to(device).unsqueeze(1)
        Mask = batch['Mask'].to(device)
        
        print(X_ji_batch.shape)

        if model.name == "ep_linear":
            outputs = model(Y_ji_batch)
        else:
            outputs = model(Y_ji_batch, N_ji_batch)

        loss = loss_fn(X_ji_batch, C_j_batch, outputs, Mask)

        if l1_lambda != 0:
            l1_norm = sum(torch.abs(p).sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
        
        loss.backward()
        optimizer.step()
        
        # calculate average loss across all batches
        total_loss += loss.item()
    avg_train_loss = total_loss / len(loader)
    
    return avg_train_loss
