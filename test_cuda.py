import numpy as np
import pandas as pd
import torch

cuda_available = torch.cuda.is_available()
print("CUDA (GPU support) is available:", cuda_available)
num_gpus = torch.cuda.device_count()
print("Number of GPUs available:", num_gpus)
tensor = torch.tensor([1, 2, 3], device='cuda:0')
is_on_gpu = tensor.is_cuda
print("Tensor is on GPU:", is_on_gpu)
