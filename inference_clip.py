import os
import torch
import hydra

from echo_diffusion.utils import instantiate_from_config


@hydra.main(config_path="configs", config_name="vae")
def main(cfg):
    model = instantiate_from_config(cfg.model)
    state_dict = torch.load("C:/Users/admin/Downloads/excel/camus-LV-MYO-crisp-55.ckpt")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    # model.load_state_dict(state_dict)
    model.eval()
    seg_z = torch.randn(1, 16)
    # seg_z = seg_z / seg_z.norm(dim=-1, keepdim=True)
    seg_recon = model.seg_decoder(seg_z)
    seg_recon = torch.softmax(seg_recon, dim=1)
    vis = seg_recon[0].detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.imshow(vis[0])
    plt.show()
    plt.imshow(vis[1])
    plt.show()
    plt.imshow(vis[2])
    plt.show()

if __name__ == '__main__':
    main()
