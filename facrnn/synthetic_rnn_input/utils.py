import os
import sys

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torcheval.metrics.functional import r2_score
from torchmetrics import Metric

sys.path.insert(1, "/".join(os.path.abspath(__file__).split("/")[0:-2]))
import disentangle


class Encoder(nn.Module):
    def __init__(
        self, obs_dim: int, n_components: int, n_total_samples: int = 1e9
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_components = n_components
        self.n_total_samples = n_total_samples
        self.register_buffer(
            "log_std",
            np.log(n_total_samples ** (-1 / (n_components + 4))) * torch.tensor(1.0),
        )
        self.linear = nn.Linear(obs_dim, n_components)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_pred_mean = self.linear(x)  # (batch_size, n_components)
        return z_pred_mean, self.log_std.expand_as(z_pred_mean)  # (batch_size, obs_dim)


class Decoder(nn.Module):
    def __init__(self, obs_dim: int, n_components: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_components = n_components
        self.linear = nn.Linear(n_components, obs_dim)
        self.log_std = nn.Parameter(-torch.ones((self.obs_dim)))

    def forward(
        self, z: torch.Tensor, x_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_pred_mean = self.linear(z)  # (batch_size, obs_dim)
        x_pred_mean = x_pred_mean + x_input
        return x_pred_mean, self.log_std[None, :].expand_as(
            x_pred_mean
        )  # (batch_size, obs_dim)


class VAE(nn.Module):
    def __init__(
        self, obs_dim: int, n_components: int, n_total_samples: int = 1e9
    ) -> None:
        super().__init__()
        self.encoder = Encoder(obs_dim, n_components, n_total_samples)
        self.decoder = Decoder(obs_dim, n_components)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, x_input):
        return self.decoder(z, x_input)


class AlignedR2Score(Metric):
    def __init__(self, n_groups_pred, n_groups_true, **kwargs):
        super().__init__(**kwargs)
        self.n_groups_pred = n_groups_pred
        self.n_groups_true = n_groups_true
        self.add_state("z_pred", default=[], dist_reduce_fx="cat")
        self.add_state("z_true", default=[], dist_reduce_fx="cat")

    def update(self, z_pred: torch.Tensor, z_true: torch.Tensor):
        if z_pred.shape != z_true.shape:
            raise ValueError(
                f"z_pred shape {z_pred.shape} and z_true shape {z_true.shape} must be the same."
            )
        self.z_pred.append(z_pred.detach().cpu())
        self.z_true.append(z_true.detach().cpu())

    def compute(self):
        z_pred = torch.cat(self.z_pred, dim=0)
        z_true = torch.cat(self.z_true, dim=0)
        group_aligned_z_pred = disentangle.align(
            z_pred,
            z_true,
            n_groups_true=self.n_groups_true,
            n_groups_pred=self.n_groups_true,
        )

        return r2_score(group_aligned_z_pred, z_true).item()


class LitVAE(L.LightningModule):
    def __init__(
        self,
        vae: VAE,
        n_groups: int,
        alpha: float = 0.0,
        beta: float = 1.0,
        n_total_samples: float = 1e7,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.vae = vae
        self.n_components = vae.encoder.n_components
        self.group_rank = int(self.n_components / n_groups)

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "latent $R^2$": AlignedR2Score(n_groups, n_groups),
            },
            prefix="train/",
        )

        self.save_hyperparameters(ignore=["vae"])

    def training_step(self, batch, batch_idx):
        x, x_convolved, z_true, x_input = batch[0], batch[1], batch[2], batch[3]
        z_pred_mean, z_pred_log_std = self.vae.encode(
            x_convolved
        )  # (batch_size, n_components)

        z = (
            torch.randn_like(z_pred_mean, device=z_pred_mean.device)
            * z_pred_log_std.exp()
            + z_pred_mean
        )

        x_pred_mean, x_pred_log_std = self.vae.decode(z, x_input)
        reconstruction_loss = (
            F.gaussian_nll_loss(
                x_pred_mean, x, x_pred_log_std.exp() ** 2, full=True, reduction="none"
            )
            .sum(dim=-1)
            .mean()
        )
        kl_loss = disentangle.kl_div_normal(z_pred_mean, z_pred_log_std).mean()
        ln_q_z, ln_prod_q_zi = disentangle.aggregated_posterior(
            z_pred_mean,
            z,
            z_pred_log_std,
            self.hparams.n_groups,
            n_total_samples=self.hparams.n_total_samples,
        )
        partial_correlation = (ln_q_z - ln_prod_q_zi).mean()

        J = (
            self.vae.encoder.linear.weight @ self.vae.decoder.linear.weight
        )  # (n_components, n_components)
        mask = 1 - torch.block_diag(
            *[
                torch.ones(self.group_rank, self.group_rank, device=J.device)
                for _ in range(self.hparams.n_groups)
            ]
        )  # (n_components, n_components)
        bd_loss = (mask * J**2).mean()

        loss = (
            reconstruction_loss
            + kl_loss
            + self.hparams.beta * partial_correlation
            + self.hparams.alpha * bd_loss
        )

        self.log_dict(
            {
                "reconstruction loss": reconstruction_loss.item(),
                "kl loss": kl_loss.item(),
                "partial correlation": partial_correlation.item(),
                "bd loss": bd_loss.item(),
                "loss": loss.item(),
            },
        )

        self.train_metrics.update(z_pred_mean, z_true)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
