# create a training script to train autoencoder
# Path: train_encoder.py

import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from echo_diffusion.utils import instantiate_from_config
from echo_diffusion.logger import ImageLogger


# using hydra to manage configs
@hydra.main(config_path='configs', config_name='autoencoder')
def main(cfg):
    pl.seed_everything(cfg.trainer.seed)
    model = instantiate_from_config(cfg.model)
    data = pl.LightningDataModule.from_datasets(
        train_dataset=instantiate_from_config(cfg.data.train),
        val_dataset=instantiate_from_config(cfg.data.train),
        test_dataset=instantiate_from_config(cfg.data.train),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    data.prepare_data()
    data.setup("fit")
    model.learning_rate = cfg.model.base_learning_rate
    trainer = pl.Trainer(
        gpus=cfg.trainer.gpus,
        max_epochs=cfg.trainer.max_epochs,
        logger=WandbLogger(),
        enable_checkpointing=True,
        resume_from_checkpoint=cfg.trainer.ckpt_path,
        callbacks=[ImageLogger(batch_frequency=1, max_images=4, clamp=True)],
    )
    trainer.fit(model, data)


if __name__ == '__main__':
    main()
