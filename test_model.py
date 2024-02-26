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
#from data_processing.bucket_loss import BucketLoss
from data_processing.load_data import setup_dataloader, read_pickle

increase_cut=0.00001
patience=5

nucleotides = ['A', 'T', 'G', 'C']
train_batch_size = 64
valid_batch_size = 1

def test_model(use_wandb, config_name, config):
        
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(cuda_available)
    print(device)
        
    train_data, valid_data, test_data = read_pickle(config["cell_type"])
    
    column_names = np.array(train_data.columns)
    feature_names = column_names[6:16]
    num_ep_features = len(feature_names)
    print(feature_names)
    num_samples = train_data.shape[0]
    num_seq_features = len(nucleotides)
    print("Number of Samples: " + str(num_samples))
    print("Number of Epigenomic Features: " + str(num_ep_features))

    torch.backends.cudnn.benchmark = True

    model = setup_model(config, device, num_ep_features, num_seq_features)
    
    train_window_size = None
    if config["train_use_sliding_window"]:
        train_window_size = config["train_window_size"]
    train_loader = setup_dataloader(train_data, feature_names, nucleotides, train_batch_size, config["train_use_sliding_window"], train_window_size)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2_lambda'])
        
    loss_fn = torch.jit.script(CustomLoss())

    # track loss curves
    loss_neural_net_train = [0] * config["epochs"]
    loss_glm_train = [0] * config["epochs"]
    
    # scheduler to reduce learning rate by half when new train loss > old train loss
    old_neural_net_train_loss = float('inf')
    epochs_no_improve = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    for epoch in range(config["epochs"]):
        print(f'Epoch {epoch+1}')
        
        neural_net_train_loss, glm_train_loss = train_epoch(model, train_loader, device, optimizer, loss_fn, config['l1_lambda'])
        loss_neural_net_train[epoch] = neural_net_train_loss
        print(f"neural net train loss: {neural_net_train_loss: .5f}")
        print(f"glm train loss: {glm_train_loss: .5f}")
        
        # early stopping
        if neural_net_train_loss < old_neural_net_train_loss - increase_cut:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve == patience:
            print("Early Stopping")
            break
        
        # reduce learning rate if new loss > old loss
        if neural_net_train_loss > old_neural_net_train_loss:
            optimizer.param_groups[0]['lr'] *= 0.5
            print(f"Reduced learning rate to {optimizer.param_groups[0]['lr']}")
            
        old_neural_net_train_loss = neural_net_train_loss
        scheduler.step(neural_net_train_loss)
        
    
    if not use_wandb:
        filename = f"./model_checkpoints/{config_name}.pth"
        torch.save(model.state_dict(), filename)
        
    return model, loss_neural_net_train
