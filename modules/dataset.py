import torch
from torch.utils.data import Dataset

class AQDataset(Dataset):
    def __init__(self, data, seq_len, horizon=4):
        self.data = data
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx : idx + self.seq_len])
        y = torch.FloatTensor([self.data[idx + self.seq_len + self.horizon - 1, 0]])
        return x, y