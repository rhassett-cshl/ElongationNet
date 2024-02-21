import torch
from torch.utils.data import Dataset

class BucketGeneDataset(Dataset):
    def __init__(self, dataframe, feature_names, nucleotides):
        self.dataframe = dataframe
        self.gene_ids = dataframe['ensembl_gene_id'].unique()
        self.genes = dataframe.groupby('ensembl_gene_id')
        self.feature_names = feature_names
        self.nucleotides = nucleotides
        self.cache = {}

    def __len__(self):
        return len(self.gene_ids)

    def __getitem__(self, idx):
        gene_id = self.gene_ids[idx]
        gene = self.genes.get_group(gene_id)
                
        if gene_id in self.cache:
            return self.cache[gene_id]
 
        result = {
            'GeneId': gene_id,
            'Seq_Name': gene['seqnames'].iloc[0],
            'Start': gene['start'],
            'End': gene['end'],
            'Strand': gene['strand'],
            
            # epigenomic features per gene j, site i
            'Y_ji':  torch.tensor(gene[feature_names].values, dtype=torch.float64),
            
            # read counts per gene j, site i
            'X_ji': torch.tensor(gene['score'].values, dtype=torch.float64),
            
            # read depth * initiation rate values per gene j
            'C_j': torch.tensor(gene['lambda_alphaj'].iloc[0], dtype=torch.float64),
            
            # GLM elongation rate predictions per gene j, site i
            'Z_ji': torch.tensor(gene['zeta'].values, dtype=torch.float64),
            
            # one-hot encoded sequences
            'N_ji': torch.tensor(gene[nucleotides].values, dtype=torch.float64), 
            'Length': len(gene)
        }
    
        self.cache[gene_id] = result

        return result