import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
import numpy as np

class BucketBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, n_buckets=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_buckets = n_buckets or 10  # Default to 10 buckets if not specified

        lengths = torch.tensor([dataset[i]['Length'] for i in range(len(dataset))], dtype=torch.float)

        # Dynamically compute bucket boundaries based on quantiles to ensure an even distribution
        boundaries = torch.quantile(lengths, torch.linspace(0, 1, steps=self.n_buckets + 1))
        self.boundaries = boundaries.unique()  # Remove duplicate boundaries

        # Update n_buckets in case there are fewer unique boundaries than requested buckets
        self.n_buckets = len(self.boundaries) - 1

        # Get bucket index for each sequence based on seq length
        self.bucket_indices = torch.bucketize(lengths, self.boundaries, right=True)

        # Efficient grouping of indices into buckets
        self.buckets = [torch.where(self.bucket_indices == i + 1)[0].tolist() for i in range(self.n_buckets)]

    def __iter__(self):
        for bucket in self.buckets:
            # Shuffle data at the bucket level
            np.random.shuffle(bucket)
            for batch in BatchSampler(torch.utils.data.SubsetRandomSampler(bucket), self.batch_size, drop_last=False):
                yield batch

    def __len__(self):
        # Include partial batch in calculation
        return sum((len(bucket) + self.batch_size - 1) // self.batch_size for bucket in self.buckets)

