import numpy as np
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from setup_model import setup_model
from train_epoch import train_epoch
from validation_epoch import valid_epoch
from loss import CustomLoss
from data_processing.load_data import setup_dataloader, read_pickle

increase_cut=0.00001
patience=5

nucleotides = ['A', 'T', 'G', 'C']
#train_batch_size = 32
valid_batch_size = 1

def train_model(use_wandb, config_name, config):
        
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
        
    train_data, valid_data, test_data = read_pickle(config["cell_type"])
    
    column_names = np.array(train_data.columns)
    feature_names = column_names[6:16]
    num_ep_features = len(feature_names)
    print(feature_names)
    num_samples = train_data.shape[0]
    num_seq_features = len(nucleotides)
    print("Number of Samples: " + str(num_samples))
    print("Number of Epigenomic Features: " + str(num_ep_features))

    model = setup_model(config, device, num_ep_features, num_seq_features)
    
    train_window_size = None
    train_stride = None
    train_batch_size = config["train_batch_size"]
    if config["train_use_sliding_window"]:
        train_window_size = config["train_window_size"]
        train_stride = config["train_stride"]
    train_loader = setup_dataloader(train_data, feature_names, nucleotides, train_batch_size, config["train_use_sliding_window"], train_window_size, train_stride)
    
    valid_window_size = None
    valid_stride = None
    if config["valid_use_sliding_window"]:
        valid_window_size = config["valid_window_size"]
        valid_stride = config["valid_stride"]
    valid_loader = setup_dataloader(valid_data, feature_names, nucleotides, valid_batch_size, config["valid_use_sliding_window"], valid_window_size, valid_stride)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2_lambda'])
        
    loss_fn = CustomLoss()

    # track loss curves
    loss_neural_net_train = [0] * config["epochs"]
    loss_neural_net_valid = [0] * config["epochs"]
    loss_glm_valid = [0] * config["epochs"]
    
    # scheduler to reduce learning rate by half when new validation loss > old validation loss
    old_neural_net_valid_loss = float('inf')
    epochs_no_improve = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    for epoch in range(config["epochs"]):
        print(f'Epoch {epoch+1}')
        
        train_loss = train_epoch(model, train_loader, device, optimizer, loss_fn, config['l1_lambda'])
        loss_neural_net_train[epoch] = train_loss
        print(f"train loss: {train_loss: .5f}")
        
        valid_neural_net_loss, valid_glm_loss, neural_net_zeta, glm_zeta = valid_epoch(model, valid_loader, device, loss_fn)
        loss_neural_net_valid[epoch] = valid_neural_net_loss
        loss_glm_valid[epoch] = valid_glm_loss
        print(f"valid neural net loss: {valid_neural_net_loss: .5f}")
        print(f"valid glm loss: {valid_glm_loss: .5f}")
        
        # compute metrics
        mae = F.l1_loss(neural_net_zeta.squeeze(), glm_zeta)
        mse = F.mse_loss(neural_net_zeta.squeeze(), glm_zeta)
        correlation_coefficient = np.corrcoef(glm_zeta, neural_net_zeta.squeeze())[0, 1]
        print("Correlation Coefficient:", correlation_coefficient)
        print(f"Mean Absolute Error: {mae.item():.4f}")
        print(f"Mean Squared Error: {mse.item():.4f}")
        
        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "valid_neural_net_loss": valid_neural_net_loss,
           "valid_glm_loss": valid_glm_loss, "correlation_coefficient": correlation_coefficient,
           "mae": mae, "mse": mse})
        
        # early stopping
        if valid_neural_net_loss < old_neural_net_valid_loss - increase_cut:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve == patience:
            print("Early Stopping")
            break
        
        # reduce learning rate if new loss > old loss
        if valid_neural_net_loss > old_neural_net_valid_loss:
            optimizer.param_groups[0]['lr'] *= 0.5
            print(f"Reduced learning rate to {optimizer.param_groups[0]['lr']}")
            
        old_neural_net_valid_loss = valid_neural_net_loss
        scheduler.step(valid_neural_net_loss)
        
    
    if not use_wandb:
        filename = f"./model_checkpoints/{config_name}.pth"
        torch.save(model.state_dict(), filename)
        
    return model, loss_neural_net_train, loss_neural_net_valid, loss_glm_valid
