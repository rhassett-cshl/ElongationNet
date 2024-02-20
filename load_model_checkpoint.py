import torch
from setup_model import setup_model

def load_model_checkpoint(config_name, config, device, num_ep_features, num_seq_features):
    
    model = setup_model(config, device, num_ep_features, num_seq_features)
    
    model.load_state_dict(torch.load(f"./model_checkpoints/{config_name}.pth", map_location=device))

    model = model.to(device)

    first_param_device = next(model.parameters()).device
    print("Model is on device:", first_param_device)

    model.double()
    
    return model