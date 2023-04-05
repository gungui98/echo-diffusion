import glob
import os

import torch
import imageio
from torch.utils.data import Dataset


class EchoDataset(Dataset):
    def __init__(self, data_dir, img_size=256):
        self.data_dir = data_dir
        self.img_size = img_size
        # glob all images in data_dir
        self.img_paths = glob.glob(os.path.join(self.data_dir, "*/*.jpg"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # force iamage to be RGB
        img = imageio.imread(img_path, pilmode="RGB")
        # resize image to img_size
        img = torch.from_numpy(img).permute(2, 0, 1).float()/255.0
        img = torch.nn.functional.interpolate(img[None], size=self.img_size,
                                              mode="bilinear", align_corners=False)[0]
        return {"image": img.permute(1, 2, 0)}

if __name__ == "__main__":
    # create a dataset
    dataset = EchoDataset(data_dir="/home/jovyan/data/camus_cityscape_format/train/images/")
    # get a sample
    sample = dataset[0]
    print(sample["image"].shape)