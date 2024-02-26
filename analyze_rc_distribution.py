import pandas as pd
import pickle

input_data_file = './data/k562_datasets.pkl'

with open(input_data_file, 'rb') as file:
	combined_datasets = pickle.load(file)
    
train_data = combined_datasets['train']

rc_description = train_data['score'].describe()

print(rc_description)
