import os

import cv2
import h5py
import matplotlib
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchmetrics.utilities.data import to_onehot

from echo_diffusion.utils import instantiate_from_config


def convert_to_torch(seg_map):
    seg_map = F.interpolate(torch.tensor(seg_map).unsqueeze(1), size=256, mode='nearest')[:, 0]
    seg_map[seg_map == 3] = 0
    seg_map = to_onehot(seg_map, 3).float()
    return seg_map


@hydra.main(config_path="configs", config_name="vae")
def main(cfg):
    # load hdf5 file

    data = h5py.File("C:/Users/admin/Downloads/excel/camus.h5", "r")
    seg_map1 = np.array(data['patient0001']['2CH']['gt'])
    seg_map2 = np.array(data['patient0002']['2CH']['gt'])

    seg_map1 = convert_to_torch(seg_map1)
    seg_map2 = convert_to_torch(seg_map2)

    seg_map = torch.cat((seg_map1[:1], seg_map2[:1]), dim=0)

    model = instantiate_from_config(cfg.model)
    state_dict = torch.load("C:/Users/admin/Downloads/excel/camus-LV-MYO-crisp-55.ckpt")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    # model.load_state_dict(state_dict)
    model.eval()
    seg_z_org = model.seg_encoder(seg_map)
    rec = []
    # create a list of 10 latent vectors that are linearly interpolated between the first and last latent vector
    for i in range(0, 100):
        seg_z = seg_z_org[:1] * (1 - i / 99) + seg_z_org[1:] * (i / 99)
        seg_z = seg_z_org[:1] + torch.randn_like(seg_z_org)
        print(seg_z.shape, seg_z.min(), seg_z.max())
        # seg_z = seg_z / seg_z.norm(dim=-1, keepdim=True)
        seg_recon = model.seg_decoder(seg_z)
        seg_recon = seg_recon.argmax(dim=1)
        vis = seg_recon[0].detach().cpu().numpy()
        rec.append(vis)

    rec = np.array(rec)
    vis = matplotlib.cm.get_cmap('viridis')(rec / rec.max())
    while True:
        for i in range(100):
            cv2.imshow('image', vis[i])
            cv2.waitKey(50)


if __name__ == '__main__':
    main()
