import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, n_sample=100, n_channel=3, img_size=64):
        self.n_sample = n_sample
        self.n_channel = n_channel
        self.img_size = img_size

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        return {"image": torch.rand(self.img_size, self.img_size, self.n_channel)}
