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


class LinearEncoder(nn.Module):
    def __init__(self, obs_dim: int, n_components: int, n_total_samples: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_components = n_components
        self.n_total_samples = n_total_samples
        self.register_buffer(
            "log_std",
            np.log(self.n_total_samples ** (-1 / (n_components + 4)))
            * torch.tensor(1.0),
        )
        self.linear = nn.Linear(obs_dim, n_components)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_pred_mean = self.linear(x)  # (batch_size, n_components)
        return z_pred_mean, self.log_std.expand_as(z_pred_mean)  # (batch_size, obs_dim)


class LinearDecoder(nn.Module):
    def __init__(self, obs_dim: int, n_components: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_components = n_components
        self.linear = nn.Linear(n_components, obs_dim)
        self.log_std = nn.Parameter(-torch.ones((self.obs_dim)))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_pred_mean = F.relu(self.linear(z))  # (batch_size, obs_dim)
        return x_pred_mean, self.log_std[None, :].expand_as(
            x_pred_mean
        )  # (batch_size, obs_dim)


class MLPEncoder(nn.Module):
    def __init__(self, obs_dim: int, n_components: int, n_total_samples: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_components = n_components
        self.n_total_samples = n_total_samples
        self.linear1 = nn.Linear(obs_dim, 32)
        self.linear2 = nn.Linear(32, n_components)
        self.register_buffer(
            "log_std",
            np.log(self.n_total_samples ** (-1 / (n_components + 4)))
            * torch.tensor(1.0),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_pred_mean = self.linear2(F.relu(self.linear1(x)))
        return z_pred_mean, self.log_std.expand_as(z_pred_mean)  # (batch_size, obs_dim)


class MLPDecoder(nn.Module):
    def __init__(self, obs_dim: int, n_components: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_components = n_components
        self.linear1 = nn.Linear(n_components, 32)
        self.linear2 = nn.Linear(32, obs_dim)
        self.log_std = nn.Parameter(-torch.ones((self.obs_dim)))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_pred_mean = F.relu(self.linear2(F.relu(self.linear1(z))))
        return x_pred_mean, self.log_std[None, :].expand_as(
            x_pred_mean
        )  # (batch_size, obs_dim)


class AlignedR2Score(Metric):
    def __init__(self, n_groups_pred, n_groups_true, **kwargs):
        super().__init__(**kwargs)
        self.n_groups_pred = n_groups_pred
        self.n_groups_true = n_groups_true
        self.add_state("z_pred", default=[], dist_reduce_fx="cat")
        self.add_state("z_true", default=[], dist_reduce_fx="cat")

    def update(self, z_pred: torch.Tensor, z_true: torch.Tensor):
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


class LitBTCVI(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        n_groups: int,
        beta: float,
        n_total_samples: int,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_groups = n_groups
        self.beta = beta
        self.n_total_samples = n_total_samples
        self.learning_rate = learning_rate

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "latent $R^2$": AlignedR2Score(2, 2),
            },
            prefix="train/",
        )

        self.save_hyperparameters(ignore=["encoder", "decoder"])

    def training_step(self, batch, batch_idx):
        x, x_convolved, z_true = batch[0], batch[1], batch[2]
        z_pred_mean, z_pred_log_std = self.encoder(
            x_convolved
        )  # (batch_size, n_components)

        z = (
            torch.randn_like(z_pred_mean, device=z_pred_mean.device)
            * z_pred_log_std.exp()
            + z_pred_mean
        )

        x_pred_mean, x_pred_log_std = self.decoder(z)
        reconstruction_loss = (
            F.gaussian_nll_loss(
                x_pred_mean,
                x,
                x_pred_log_std.exp() ** 2,
                full=True,
                reduction="none",
            )
            .sum(dim=-1)
            .mean()
        )
        kl_loss = disentangle.kl_div_normal(z_pred_mean, z_pred_log_std).mean()
        ln_q_z, ln_prod_q_zi = disentangle.aggregated_posterior(
            z_pred_mean,
            z,
            z_pred_log_std,
            self.n_groups,
            n_total_samples=self.n_total_samples,
        )
        partial_correlation = (ln_q_z - ln_prod_q_zi).mean()

        loss = reconstruction_loss + kl_loss + self.beta * partial_correlation

        self.log_dict(
            {
                "reconstruction loss": reconstruction_loss.item(),
                "kl loss": kl_loss.item(),
                "partial correlation": partial_correlation.item(),
                "loss": loss.item(),
            },
        )

        self.train_metrics.update(z_pred_mean, z_true)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
