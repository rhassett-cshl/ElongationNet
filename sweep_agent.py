import wandb
from train_model import train_model

def sweep_agent():
    with wandb.init() as run:
        config = run.config
        config_name = run.project
        train_model(True, config_name, config) 
