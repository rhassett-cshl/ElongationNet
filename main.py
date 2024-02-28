import argparse
import json
import wandb
import torch
from train_model import train_model
from train_model_performance_analysis import train_model_performance_analysis
from save_results import save_results
from sweep_agent import sweep_agent


def main():
    parser = argparse.ArgumentParser(description="Elongation Net")
    parser.add_argument("--mode", choices=['train', 'sweep', 'save_results', 'performance_analysis'], required=True, help="Operation mode: train, sweep, save_results, or performance_analysis")
    parser.add_argument("--config_name", required=True, help="Config name: Name of config file")

    args = parser.parse_args()
        
    with open("./configs/" + args.config_name + ".json", 'r') as file:
        config = json.load(file)
    
    if args.mode == 'train':
        train_model(False, args.config_name, config)
    elif args.mode == 'sweep':

        sweep_config = {
            'method': 'grid'
        }
        metric = {
            'name': 'valid_neural_net_loss',
            'goal': 'minimize'   
        }
        sweep_config['metric'] = metric
        sweep_config['parameters'] = config
        
        sweep_id = wandb.sweep(sweep_config, project=args.config_name)
        wandb.agent(sweep_id, function=sweep_agent)
        
    elif args.mode == 'save_results':
        save_results(args.config_name, config)
    elif args.mode == 'performance_analysis':
        train_model_performance_analysis(False, args.config_name, config)
    else:
        print("Invalid mode selected. Please choose from 'train', 'sweep', or 'analyze'.")

    

if __name__ == "__main__":
    main()
