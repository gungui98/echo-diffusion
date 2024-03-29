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
        img = imageio.imread(img_path, pilmode="L")
        mask = imageio.imread(segmap_path, pilmode="L")
        # resize image to img_size
        img = torch.from_numpy(img).float() / 255.0
        mask = torch.from_numpy(mask).float()

        img = torch.nn.functional.interpolate(img[None, None], size=self.img_size,
                                              mode="bilinear", align_corners=False)[0, 0]
        mask = torch.nn.functional.interpolate(mask[None, None], size=self.img_size,
                                               mode="nearest")[0, 0].long()
        mask[mask == 3] = 0
        return {"image": img, "segmap": mask}


if __name__ == "__main__":
    # create a dataset
    dataset = EchoDataset(data_dir="/home/jovyan/data/camus_cityscape_format/train/images/")
    # get a sample
    sample = dataset[0]
    print(sample["image"].shape)
