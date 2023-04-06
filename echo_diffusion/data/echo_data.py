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
        self.segmap_paths = [p.replace("images", "seg_maps").replace(".jpg", ".png")
                             for p in self.img_paths]
        self.data = list(zip(self.img_paths, self.segmap_paths))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        segmap_path = self.segmap_paths[idx]
        # force iamage to be RGB
        img = imageio.imread(img_path, pilmode="RGB")
        mask = imageio.imread(segmap_path, pilmode="L")
        # resize image to img_size
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        img = torch.nn.functional.interpolate(img[None], size=self.img_size,
                                              mode="bilinear", align_corners=False)[0]
        mask = torch.nn.functional.interpolate(mask[None], size=self.img_size,
                                               mode="nearest")[0]
        return {"image": img.permute(1, 2, 0), "mask": mask}


if __name__ == "__main__":
    # create a dataset
    dataset = EchoDataset(data_dir="/home/jovyan/data/camus_cityscape_format/train/images/")
    # get a sample
    sample = dataset[0]
    print(sample["image"].shape)
