#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

if len(sys.argv) == 2:
    cell_type = sys.argv[1]
    print(f"Received argument: {cell_type}")
else:
    print("Error: This script requires exactly one argument.")
    sys.exit(1)  


# In[ ]:


import numpy as np
import pandas as pd

# File path for saving
filename = f'./data/{cell_type}_datasets.pkl'

froot = f'./data/{cell_type}_epAllmer_zeta_norm.csv'
df = pd.read_csv(froot)

print(df.head())


# In[ ]:





# In[38]:


from sklearn.model_selection import train_test_split

# train size = 80%, validation size = 10%, test size = 10%
train_size = 0.8

grouped = df.groupby('ensembl_gene_id')

# split by gene into train, val, test sets
train_idx, temp_idx = train_test_split(list(grouped.groups.keys()), test_size=(1.0 - train_size), random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

# create dictionary mapping each gene id to its assigned train, val, test dataset labels
dataset_mapping = {gene_id: 'train' for gene_id in train_idx}
dataset_mapping.update({gene_id: 'val' for gene_id in val_idx})
dataset_mapping.update({gene_id: 'test' for gene_id in test_idx})

# filter rows based on assigned dataset field
df['dataset'] = df['ensembl_gene_id'].map(dataset_mapping)
train_data = df[df['dataset'] == 'train']
valid_data = df[df['dataset'] == 'val']
test_data = df[df['dataset'] == 'test']

print("train data size: " + str(len(train_data)))
print("val data size: " + str(len(valid_data)))
print("test data size: " + str(len(test_data)) + "\n")


# In[39]:


print(train_data.iloc[0])


# In[40]:


print("train # genes: " + str(len(train_data.groupby('ensembl_gene_id'))))
print("val # genes: " + str(len(valid_data.groupby('ensembl_gene_id'))))
print("test # genes: " + str(len(test_data.groupby('ensembl_gene_id'))))


# In[41]:


combined_datasets = {
    'train': train_data,
    'valid': valid_data,
    'test': test_data
}

# Serialize the combined datasets to a pickle file with protocol=4 or higher
with open(filename, 'wb') as file:
    pickle.dump(combined_datasets, file, protocol=pickle.HIGHEST_PROTOCOL) # protocol 4 for python >= 3.4


# In[42]:


with open(file_path, 'rb') as file:
    combined_datasets = pickle.load(file)

dataset1 = combined_datasets['train']

print(dataset1.iloc[0])


# In[ ]:




