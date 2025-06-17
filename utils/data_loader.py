import os
import torch
from torch.utils.data import Dataset
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        label = int(os.path.basename(self.files[idx]).split("_")[0])  # e.g. "1_sample.npy"
        return torch.tensor(data, dtype=torch.float32), label
