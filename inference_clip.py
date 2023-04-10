import os
import torch
import hydra

from echo_diffusion.utils import instantiate_from_config

@hydra.main(config_path="configs", config_name="vae")
def main(cfg):
    model = instantiate_from_config(cfg.model)
    state_dict = torch.load("/home/jovyan/code/uncertainty/outputs/2022-10-19/01-35-40/log/55/camus-LV-MYO-crisp-55.ckpt")
    model.load_state_dict(state_dict)

if __name__ == '__main__':
    main()