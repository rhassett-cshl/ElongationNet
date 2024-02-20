import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import BatchSampler

""" Batch subsequences with same gene id together """
class GeneBatchSampler(BatchSampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.batches = self._create_batches()

    def _create_batches(self):
        # Group indices by GeneId
        gene_id_to_indices = {}
        for idx in range(len(self.dataset)):
            gene_id = self.dataset[idx]['GeneId']
            if gene_id not in gene_id_to_indices:
                gene_id_to_indices[gene_id] = []
            gene_id_to_indices[gene_id].append(idx)

        return list(gene_id_to_indices.values())

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
