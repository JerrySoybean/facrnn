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

method_list = [
    (1, 2),
    "(1, 2)_LINT",
    (2, 1),
    (1, 4),
    "(1, 4)_LINT",
    (2, 2),
    (4, 1),
    # (1, 8),
    # "(1, 8)_LINT",
    # (2, 4),
    # (4, 2),
    # (8, 1),
    "MLP",
]
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
data_dict = pd.read_pickle("data/source_data_array_with_dir.pkl")
x = torch.tensor(data_dict["neural"]).to(torch.float32)
vel = torch.tensor(data_dict["vel"]).to(torch.float32)
loc = vel.cumsum(dim=1)
n_trials, n_time_bins, n_neurons = x.shape
# x_convolved = torch.tanh(torch.cat([x[:, 0:1], x[:, :-1]], dim=1))
# x_convolved = torch.cat([x[:, 0:1], x[:, :-1]], dim=1)
x_convolved = x.clone()
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

df_result = pd.DataFrame(
    index=np.arange(1),
    columns=[
        "radian",
        "latent $R^2$",
    ],
)

tag = "conv"
results_file = f"results_{tag}"

with torch.no_grad():
    z_pred_mean = torch.load(f"{results_file}/{name}_z_aligned.pt")
    n_components = z_pred_mean.shape[-1]

    for idx, theta in enumerate(np.linspace(0, np.pi / 2, 13)):

        loc_rot = (
            loc_reshape
            @ torch.from_numpy(
                np.array(
                    [
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)],
                    ]
                )
            ).float()
        )

        z_aligned_reshape = disentangle.align(
            z_pred_mean.reshape(
                n_trials * n_time_bins,
                n_components,
            ),
            loc_rot,
            n_groups_true=2,
            n_groups_pred=2,
        )  # (n_trials * n_time_bins, 2)
        z_aligned = z_aligned_reshape.reshape(
            n_trials, n_time_bins, -1
        )  # (n_trials, n_time_bins, 2)

        df_result.at[idx, "radian"] = theta
        df_result.at[idx, "latent $R^2$"] = r2_score(
            z_aligned_reshape,
            loc_rot,
        ).item()

df_result.to_csv(f"{results_file}/{name}_rotation.csv", index=False)
