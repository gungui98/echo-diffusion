import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, n_sample=100, image_channel=3, segmap_channel=4, img_size=64):
        self.n_sample = n_sample
        self.image_channel = image_channel
        self.segmap_channel = segmap_channel
        self.img_size = img_size

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        image = torch.rand(self.img_size, self.img_size, self.image_channel)
        segmap = torch.randint(0, 1, (self.img_size, self.img_size))
        return {"image": image, "segmap": segmap}


if __name__ == '__main__':
    dataset = DummyDataset()
    print(dataset[0]["image"].shape)
    print(dataset[0]["segmap"].shape)
