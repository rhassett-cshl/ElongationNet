import torch
from torch.utils.data import BatchSampler

class BucketBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, bucket_size=2000):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size

        lengths = torch.tensor([dataset[i]['Length'] for i in range(len(dataset))])
        max_length = lengths.max_length().item()
        min_length = lengths.min().item()
        # calculate number of buckets from min seq length to max seq length with size bucket_size
        self.n_buckets = ((max_length - min_length) // 2000) + 1
        self.boundaries = torch.arange(min_length, max_length + 2000, step=2000)
        # get bucket index for each sequence based on seq length
        self.bucket_indices = torch.bucketize(lengths, self.boundaries, right=True)

        # Efficient grouping of indices into buckets using PyTorch operations
        self.buckets = [torch.where(self.bucket_indices == i + 1)[0].tolist() for i in range(self.n_buckets)]

    def __iter__(self):
        for bucket in self.buckets:
            for batch in BatchSampler(torch.utils.data.SubsetRandomSampler(bucket), self.batch_size, drop_last=False):
                yield batch

    # calculate number of batches created
    def __len__(self):
        # Include partial batch in calculation
        return sum((len(bucket) + self.batch_size - 1) // self.batch_size for bucket in self.buckets)
