import itertools
import random
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from torchmetrics.utilities.data import to_onehot
from monai.losses import DiceLoss
from tqdm import tqdm

from echo_diffusion.losses.dice_loss import DiceLoss
from echo_diffusion.models.decoder import Decoder
from echo_diffusion.models.encoder import Encoder
import numpy as np
import einops
import matplotlib



class EchoVAE(pl.LightningModule):
    def __init__(self,
                 encoder_img_params=None,
                 encoder_seg_params=None,
                 decoder_img_params=None,
                 decoder_seg_params=None,
                 image_key="image",
                 segmap_key="segmap",
                 loss_params=None,
                 ):
        super().__init__()


        self.segmap_key = segmap_key
        self.image_key = image_key
        self.img_encoder = Encoder(**encoder_img_params)
        self.seg_encoder = Encoder(**encoder_seg_params)
        self.img_decoder = Decoder(**decoder_img_params)
        self.seg_decoder = Decoder(**decoder_seg_params)
        self.output_distribution = self.img_encoder.output_distribution and \
                                   self.seg_encoder.output_distribution

        # project latent space into same dimension, using to train clip model
        img_latent_size = 64
        seg_latent_size = 16
        latent_size = 8
        self.img_proj = nn.Linear(img_latent_size, latent_size)
        self.seg_proj = nn.Linear(seg_latent_size, latent_size)

        self.logit_scale = torch.nn.Parameter(torch.randn(1), requires_grad=True)
        self.register_parameter('logit_scale', self.logit_scale)
        self.is_val_step = False
        self.learning_rate = 1e-3

        self.loss = loss_params
        self.interpolation_augmentation_samples = 0
        self.linear_constraint_weight = 0

        self.decode_img = False
        self.decode_seg = True

        self._dice = DiceLoss()
        self.img_reconstruction_loss = nn.MSELoss()
        self.attr_reg = False

    def encode_dataset(self, datamodule, progress_bar: bool = False):
        """Encodes masks from the train and val sets of a dataset in the latent space learned by an CLIP model.

        Args:
            system: CLIP system used to encode masks in a latent space.
            datamodule: Abstraction of the dataset to encode, allowing to access both the training and validation sets.
            progress_bar: Whether to display a progress bar for the encoding of the samples.

        Returns:
            Array of training and validation samples encoded in the latent space.
        """
        # Setup the datamodule used to get the data points to encode in the latent space
        datamodule.setup(stage="fit")
        train_dataloader, val_dataloader = datamodule.train_dataloader(), datamodule.val_dataloader()
        data = itertools.chain(train_dataloader, val_dataloader)
        num_batches = len(train_dataloader) + len(val_dataloader)

        if progress_bar:
            data = tqdm(data, desc="Encoding groundtruths", total=num_batches, unit="batch")

        # Encode training and validation groundtruths in the latent space
        dataset_samples = []
        with torch.no_grad():
            for batch in data:
                seg = batch[self.segmap_key].to(self.device)
                if datamodule.data_params.out_shape[0] > 1:
                    seg = to_onehot(seg, num_classes=datamodule.data_params.out_shape[0]).float()
                else:
                    seg = seg.unsqueeze(1).float()
                dataset_samples.append(self.seg_encoder(seg).cpu())

        dataset_samples = torch.cat(dataset_samples).numpy()

        return dataset_samples

    def encode_dataloader(self, dataloader, output_shape):
        dataset_samples = []
        with torch.no_grad():
            for batch in dataloader:
                seg = batch[self.segmap_key].to(self.device)
                if output_shape[0] > 1:
                    seg = to_onehot(seg, num_classes=output_shape[0]).float()
                else:
                    seg = seg.unsqueeze(1).float()
                dataset_samples.append(self.seg_encoder(seg).cpu())

        dataset_samples = torch.cat(dataset_samples).numpy()
        return dataset_samples

    def decode(self, encoding: np.ndarray) -> np.ndarray:
        """Decodes a sample, or batch of samples, from the latent space to the output space.

        Args:
            system: Autoencoder system with generative capabilities used to decode the encoded samples.
            encoding: Sample, or batch of samples, from the latent space to decode.

        Returns:
            Decoded sample, or batch of samples.
        """
        encoding = encoding.astype(np.float32)  # Ensure the encoding is in single-precision float dtype
        if len(encoding.shape) == 1:
            # If the input isn't a batch of value, add the batch dimension
            encoding = encoding[None, :]
        encoding_tensor = torch.from_numpy(encoding)
        decoded_sample = self.seg_decoder(encoding_tensor)
        decoded_sample = decoded_sample.argmax(1) if decoded_sample.shape[1] > 1 else torch.sigmoid(
            decoded_sample).round()

        return decoded_sample.squeeze().cpu().detach().numpy()

    def training_step(self, batch, batch_nb):
        self.is_val_step = False
        loss_dict = self.trainval_step(batch, batch_nb)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_dict

    def validation_step(self, batch, batch_nb):
        self.is_val_step = True
        loss_dict = self.trainval_step(batch, batch_nb)
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_dict

    def trainval_step(self, batch: Any, batch_nb: int):
        img, seg = batch[self.image_key], batch[self.segmap_key]
        # check if input is correct shape
        if img.shape[1] != self.img_encoder.in_channels:
            img = einops.rearrange(img, 'b h w c -> b c h w')
        seg_onehot = to_onehot(seg, num_classes=self.seg_encoder.in_channels).float()
        logs = {}
        batch_size = img.shape[0]

        if self.output_distribution:
            img_mu, img_logvar = self.img_encoder(img)
            seg_mu, seg_logvar = self.seg_encoder(seg_onehot)
        else:
            img_mu = self.img_encoder(img)
            seg_mu = self.seg_encoder(seg_onehot)

        if self.interpolation_augmentation_samples > 0 and not self.is_val_step:
            augmented_samples = []
            for i in range(self.interpolation_augmentation_samples // 2):
                i1, i2 = random.randrange(len(img)), random.randrange(len(img))
                aug_seg1 = torch.lerp(seg_mu[i1], seg_mu[i2], random.uniform(-0.5, -1))
                aug_seg2 = torch.lerp(seg_mu[i1], seg_mu[i2], random.uniform(1.5, 2))
                augmented_samples.extend([aug_seg1[None], aug_seg2[None]])

            augmented_samples = torch.cat(augmented_samples, dim=0)
            augmentated_seg_mu = torch.cat([seg_mu, augmented_samples], dim=0)
        else:
            augmentated_seg_mu = seg_mu

        # Compute CLIP loss
        img_logits, seg_logits = self.clip_forward(img_mu, augmentated_seg_mu)

        labels = torch.arange(batch_size, device=self.device)
        loss_i = F.cross_entropy(img_logits, labels, reduction='none')
        loss_t = F.cross_entropy(seg_logits[:batch_size], labels, reduction='none')

        loss_i = loss_i.mean()
        loss_t = loss_t.mean()
        clip_loss = (loss_i + loss_t) / 2

        img_accuracy = img_logits.argmax(0)[:batch_size].eq(labels).float().mean()
        seg_accuracy = seg_logits.argmax(0).eq(labels).float().mean()

        loss = 0
        loss += self.loss.clip_weight * clip_loss

        if self.linear_constraint_weight:
            regression_target = batch[self.linear_constraint_attr]
            regression = self.regression_module(seg_mu)
            regression_mse = F.mse_loss(regression, regression_target)
            loss += self.linear_constraint_weight * regression_mse
            logs.update({"regression_mse": regression_mse})

        # Compute VAE loss
        if self.decode_seg:
            if self.output_distribution:
                seg_z = self.reparameterize(seg_mu, seg_logvar)
                seg_kld = self.latent_space_metrics(seg_mu, seg_logvar)
            else:
                seg_kld = 0
                seg_z = seg_mu

            seg_recon = self.seg_decoder(seg_z)

            seg_metrics = self.seg_reconstruction_metrics(seg_recon, seg)

            seg_vae_loss = self.loss.reconstruction_weight * seg_metrics['seg_recon_loss'] + \
                           self.loss.kl_weight * seg_kld

            logs.update({'seg_vae_loss': seg_vae_loss, 'seg_kld': seg_kld})
            logs.update(seg_metrics)

            if self.is_val_step and batch_nb == 0:
                seg_recon = seg_recon.argmax(1) if seg_recon.shape[1] > 1 else torch.sigmoid(seg_recon).round()
                self.log_images(title='Sample (seg)', num_images=5,
                                img_dict={'Image': img.cpu().squeeze().numpy(),
                                              'GT': seg.cpu().squeeze().numpy(),
                                              'Pred': seg_recon.squeeze().detach().cpu().numpy()})

            loss += seg_vae_loss

        if self.attr_reg:
            attr_metrics = self._compute_latent_space_metrics(seg_mu, batch)
            attr_reg_sum = sum(attr_metrics[f"{attr}_attr_reg"] for attr in
                               CamusTags.list_available_attrs(self.data_params.labels))
            loss += attr_reg_sum * 10
            logs.update({'attr_reg_loss': attr_reg_sum})

        if self.decode_img:
            if self.output_distribution:
                img_z = self.reparameterize(img_mu, img_logvar)
                img_kld = self.latent_space_metrics(img_mu, img_logvar)
            else:
                img_kld = 0
                img_z = img_mu

            img_recon = self.img_decoder(img_z)
            img_metrics = self.img_reconstruction_metrics(img_recon, img)

            img_vae_loss = self.loss.reconstruction_weight * img_metrics['img_recon_loss'] + \
                           self.loss.kl_weight * img_kld

            logs.update({'img_vae_loss': img_vae_loss, 'img_kld': img_kld, })
            logs.update(img_metrics)

            if self.is_val_step and batch_nb == 0:
                self.log_images(title='Sample (img)', num_images=5,
                                img_dict={'Image': img.cpu().squeeze().numpy(),
                                              'Pred': img_recon.squeeze().detach().cpu().numpy()})

            loss += img_vae_loss

        logs.update({
            'loss': loss,
            'clip_loss': clip_loss,
            'img_accuracy': img_accuracy,
            'seg_accuracy': seg_accuracy,
        })

        return logs

    def clip_forward(self, image_features, seg_features):
        image_features = self.img_proj(image_features)
        seg_features = self.seg_proj(seg_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        seg_features = seg_features / seg_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ seg_features.t()
        logits_per_seg = logit_scale * seg_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_seg

    def log_images(self, title, num_images, img_dict):
        # TODO: move to image logger callback
        for k in img_dict.keys():
            if len(img_dict[k].shape) == 4: # if images is gt segment mask then convert it into viridis color space
                images = einops.rearrange(img_dict[k], 'b c h w -> b h w c')
            else:
                images = matplotlib.cm.get_cmap('viridis')(img_dict[k]/img_dict[k].max())[..., :3]
            vis_image = np.vstack(images[0:num_images])
            self.logger.experiment.log({
                title: wandb.Image(vis_image)})

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def seg_reconstruction_metrics(self, recon_x: torch.Tensor, x: torch.Tensor):
        # Segmentation accuracy metrics
        if recon_x.shape[1] == 1:
            ce = F.binary_cross_entropy_with_logits(recon_x.squeeze(), x.type_as(recon_x))
        else:
            ce = F.cross_entropy(recon_x, x)

        dice_values = self._dice(recon_x, x)
        # TODO: calculate mean dice for each class
        mean_dice = dice_values.mean()

        loss = (self.loss.cross_entropy_weight * ce) + (self.loss.dice_weight * (1 - mean_dice))
        return {"seg_recon_loss": loss, "seg_ce": ce, "dice": mean_dice}

    def img_reconstruction_metrics(self, recon_x: torch.Tensor, x: torch.Tensor):
        return {"img_recon_loss": self.img_reconstruction_loss(recon_x, x)}

    def latent_space_metrics(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld

    def _compute_latent_space_metrics(self, mu, batch):
        """Computes metrics on the input's encoding in the latent space.
        Adds the attribute regularization term to the loss already computed by the parent's implementation.
        Args:
            out: Output of a forward pass with the autoencoder network.
            batch: Content of the batch of data returned by the dataloader.
        References:
            - Computation of the attribute regularization term inspired by the original paper's implementation:
              https://github.com/ashispati/ar-vae/blob/master/utils/trainer.py#L378-L403
        Returns:
            Metrics useful for computing the loss and tracking the system's training progress:
                - metrics computed by ``super()._compute_latent_space_metrics``
                - attribute regularization term for each attribute (under the "{attr}_attr_reg" label format)
        """
        attr_metrics = {}
        for attr_idx, attr in enumerate(CamusTags.list_available_attrs(self.data_params.labels)):
            # Extract dimension to regularize and target for the current attribute
            latent_code = mu[:, attr_idx].unsqueeze(1)
            attribute = batch[attr]

            # Compute latent distance matrix
            latent_code = latent_code.repeat(1, latent_code.shape[0])
            lc_dist_mat = latent_code - latent_code.transpose(1, 0)

            # Compute attribute distance matrix
            attribute = attribute.repeat(1, attribute.shape[0])
            attribute_dist_mat = attribute - attribute.transpose(1, 0)

            # Compute regularization loss
            # lc_tanh = torch.tanh(lc_dist_mat * self.delta)
            lc_tanh = torch.tanh(lc_dist_mat * 1)
            attribute_sign = torch.sign(attribute_dist_mat)
            attr_metrics[f"{attr}_attr_reg"] = F.l1_loss(lc_tanh, attribute_sign)

        return attr_metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def encode_set(self, dataloader):
        train_set_features = []
        train_set_segs = []
        for batch in tqdm(iter(dataloader)):
            seg = batch[self.segmap_key].to(self.device)
            train_set_segs.append(seg.cpu())
            if self.data_params.out_shape[0] > 1:
                seg = to_onehot(seg, num_classes=self.data_params.out_shape[0]).float()
            else:
                seg = seg.unsqueeze(1).float()
            seg_mu = self.seg_encoder(seg)
            train_set_features.append(seg_mu.detach().cpu())
        return train_set_features, train_set_segs

    # def on_fit_end(self) -> None:
    #     if self.save_samples:
    #         print("Generate train features")
    #
    #         datamodule = self.trainer.datamodule
    #
    #         train_set_features, train_set_segs = self.encode_set(datamodule.train_dataloader())
    #         val_set_features, val_set_segs = self.encode_set(datamodule.val_dataloader())
    #         train_set_features.extend(val_set_features)
    #         train_set_segs.extend(val_set_segs)
    #
    #         self.train_set_features = torch.cat(train_set_features)
    #         self.train_set_segs = torch.cat(train_set_segs)
    #         Path(self.save_samples).parent.mkdir(exist_ok=True)
    #         torch.save({'features': self.train_set_features,
    #                     'segmentations': self.train_set_segs}, self.save_samples)


if __name__ == '__main__':
    from echo_diffusion.data import DummyDataset
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
    configs = OmegaConf.load('configs/vae.yaml')
    model = EchoVAE(encoder_img_params=configs.model.encoder_img,
                    encoder_seg_params=configs.model.encoder_seg,
                    decoder_img_params=configs.model.decoder_img,
                    decoder_seg_params=configs.model.decoder_seg,
                    )
