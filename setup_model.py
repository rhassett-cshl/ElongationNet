import torch
from models.Ep_Linear import Ep_Linear
from models.Ep_Allmer_Linear import Ep_Allmer_Linear
from models.Ep_Allmer_CNN import Ep_Allmer_CNN
from models.Ep_Allmer_CNN_LSTM import Ep_Allmer_CNN_LSTM
from models.Ep_Allmer_LSTM import Ep_Allmer_LSTM

def setup_model(config, device, num_ep_features, num_seq_features):
    
    cuda_available = torch.cuda.is_available()
    # num_gpus = torch.cuda.device_count()
    # print("Number of GPUs available:", num_gpus)        

    if config["model_type"] == 'ep_linear':
        model = Ep_Linear(num_ep_features)
    elif config["model_type"] == 'ep_seq_linear':
        model = Ep_Allmer_Linear(num_ep_features, num_seq_features)
    elif config["model_type"] == 'lstm':
        model = Ep_Allmer_LSTM(num_ep_features + num_seq_features)
    elif config["model_type"] == 'cnn':
        model = Ep_Allmer_CNN(num_ep_features, num_seq_features, 
                                config["y_channels"], config["y_kernel_sizes"], 
                                config["n_channels"], config["n_kernel_sizes"], config["dropout"])
    elif config["model_type"] == 'cnn_lstm':
        model = Ep_Allmer_CNN_LSTM(num_ep_features, num_seq_features, 
                                   config["y_channels"], config["y_kernel_sizes"], 
                                   config["n_channels"], config["n_kernel_sizes"], config["dropout"], 
                                   config["num_lstm_layers"], config["lstm_layer_size"], config["bidirectional"])
        

    if cuda_available:
        """
        if num_gpus > 1:
            print("Using", num_gpus, "GPUs")
            model = torch.nn.DataParallel(model)
        """
        model = model.to('cuda')
    
    print(model)
    
    model.double()
    
    return model.to(device)
    
    