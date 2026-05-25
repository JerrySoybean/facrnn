import argparse
import os
import sys

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from pdisvae import evaluate, kl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import mean_squared_error, r2_score

sys.path.insert(1, "/".join(os.path.abspath(__file__).split("/")[0:-2]))
import disentangle
import utils

torch.set_float32_matmul_precision("high")

## arguments
parser = argparse.ArgumentParser()
parser.add_argument("idx", type=int)
args = parser.parse_args()

# method_list = [(1, 2), (2, 1), (1, 8), (2, 4), (4, 2), (8, 1), "MLP"]
method_list = [(1, 2), (2, 1), (1, 4), (2, 2), (4, 1), "MLP"]
seed_list = np.arange(5)

arg_index = np.unravel_index(
    args.idx,
    (
        len(method_list),
        len(seed_list),
    ),
)
method, seed = (
    method_list[arg_index[0]],
    seed_list[arg_index[1]],
)

print(f"method: {method}")
print(f"seed: {seed}")
name = f"{method}_{seed}"


## data
data_dict = pd.read_pickle("data/source_data_array_with_dir_2.pkl")
x = torch.tensor(data_dict["neural"]).to(torch.float32)
vel = torch.tensor(data_dict["vel"]).to(torch.float32)
loc = vel.cumsum(dim=1)
n_trials, n_time_bins, n_neurons = x.shape
x_convolved = torch.tanh(torch.cat([x[:, 0:1], x[:, :-1]], dim=1))
x_convolved_reshape = x_convolved.reshape(
    n_trials * n_time_bins,
    n_neurons,
)  # (n_trials * n_time_bins, n_neurons)
x_reshape = x.reshape(
    n_trials * n_time_bins,
    n_neurons,
)  # (n_trials * n_time_bins, n_neurons)
vel_reshape = vel.reshape(
    n_trials * n_time_bins,
    2,
)  # (n_trials * n_time_bins, 2)
loc_reshape = loc.reshape(
    n_trials * n_time_bins,
    2,
)  # (n_trials * n_time_bins, 2)
dataloader = DataLoader(
    TensorDataset(x_reshape, x_convolved_reshape, vel_reshape),
    batch_size=128,
    shuffle=True,
)

if method == "MLP":
    n_groups = 2
    group_rank = 1
else:
    n_groups, group_rank = method
n_components = n_groups * group_rank
n_total_samples = 512


## initialization
torch.manual_seed(seed)
if method == "MLP":
    encoder = utils.MLPEncoder(
        obs_dim=n_neurons,
        n_components=n_components,
        n_total_samples=n_total_samples,
    )
    decoder = utils.MLPDecoder(
        obs_dim=n_neurons,
        n_components=n_components,
    )
else:
    encoder = utils.LinearEncoder(
        obs_dim=n_neurons,
        n_components=n_components,
        n_total_samples=n_total_samples,
    )
    decoder = utils.LinearDecoder(
        obs_dim=n_neurons,
        n_components=n_components,
    )


## Lightning module
tag = "conv"
results_file = f"results_{tag}"

n_epochs = 1000

learning_rate = 1e-3

lit_btcvi = utils.LitBTCVI(
    encoder,
    decoder,
    n_groups=n_groups,
    beta=2,
    n_total_samples=n_total_samples,
    learning_rate=learning_rate,
)

wandb_logger = WandbLogger(
    name=name,
    project=f"neurips2025-{__file__.split('/')[-2]}",
    save_dir=results_file,
    tags=[tag],
)
trainer = L.Trainer(
    logger=wandb_logger,
    min_epochs=n_epochs,
    max_epochs=n_epochs,
    log_every_n_steps=1,
    enable_progress_bar=True,
    devices=1,
    accelerator="gpu",
)
trainer.fit(
    model=lit_btcvi,
    train_dataloaders=dataloader,
)
trainer.save_checkpoint(f"{results_file}/{name}.ckpt")
# lit_btcvi = utils.LitBTCVI.load_from_checkpoint(
#     f"{results_file}/{name}.ckpt", encoder=encoder, decoder=decoder
# )


## evaluation
n_monte_carlo = 10
df_result = pd.DataFrame(
    index=np.arange(1),
    columns=[
        "conditional log-likelihood",
        "reconstruction $R^2$",
        "reconstruction RMSE",
        "partial correlation",
        "total correlation",
        "latent $R^2$",
    ],
)
with torch.no_grad():
    z_pred_mean, z_pred_log_std = encoder.forward(
        x_convolved_reshape
    )  # (n_samples, n_components)

    x_pred_mean, x_pred_log_std = decoder.forward(z_pred_mean)

    df_result.at[0, "conditional log-likelihood"] = (
        -F.gaussian_nll_loss(
            x_pred_mean,
            x_reshape,
            x_pred_log_std.exp() ** 2,
            full=True,
            reduction="none",
        )
        .sum(dim=-1)
        .mean()
        .item()
    )

    df_result.at[0, "reconstruction $R^2$"] = r2_score(x_pred_mean, x_reshape).item()
    df_result.at[0, "reconstruction RMSE"] = (
        mean_squared_error(x_pred_mean, x_reshape).item() ** 0.5
    )

    df_result.at[0, "partial correlation"] = evaluate.partial_correlation(
        z_pred_mean,
        z_pred_log_std,
        n_groups=2,
        n_monte_carlo=n_monte_carlo,
    ).item()
    df_result.at[0, "total correlation"] = evaluate.partial_correlation(
        z_pred_mean,
        z_pred_log_std,
        n_groups=n_components,
        n_monte_carlo=n_monte_carlo,
    ).item()

    z_aligned_reshape = disentangle.align(
        z_pred_mean.reshape(n_trials, n_time_bins, n_components)
        .cumsum(dim=1)
        .reshape(-1, n_components),
        loc.reshape(n_trials * n_time_bins, -1),
        n_groups_true=2,
        n_groups_pred=2,
    )  # (n_trials * n_time_bins, 2)
    z_aligned = z_aligned_reshape.reshape(
        n_trials, n_time_bins, -1
    )  # (n_trials, n_time_bins, 2)

    torch.save(
        z_aligned,
        f"{results_file}/{name}_z_aligned.pt",
    )
    df_result.at[0, "latent $R^2$"] = r2_score(
        z_aligned_reshape,
        loc_reshape,
    ).item()

df_result.to_csv(f"{results_file}/{name}.csv", index=False)
