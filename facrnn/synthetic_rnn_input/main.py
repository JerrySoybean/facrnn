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
from sklearn.decomposition import FastICA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import r2_score

sys.path.insert(1, "/".join(os.path.abspath(__file__).split("/")[0:-2]))
print(sys.path)
import disentangle
import utils

torch.set_float32_matmul_precision("high")

## arguments
parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int)
parser.add_argument(
    "--mode", type=str, choices=["train", "eval", "both"], default="both"
)
args = parser.parse_args()

method_list = [
    "lrRNN",
    "lrRNN+ICA",
    "LINT",
    "bdRNN",
    "DisRNN-full",
    "DisRNN",
]
seed_list = np.arange(10)

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


## hyperparameters
n_groups_true = 2
group_rank_true = 3


## data
trial = 0
df_data = pd.read_pickle("data/data.pkl")
x = df_data.at[trial, "x"]
z = df_data.at[trial, "z"]
x_input = df_data.at[trial, "x_input"]
kernel_size = 1
kernel = torch.exp(torch.arange(kernel_size).float())
kernel = kernel / kernel.sum()
x_convolved = F.conv1d(
    F.pad(torch.tanh(x.T), (kernel_size, -1), "constant", 0)[:, None, :],
    kernel[None, None, :],
)[
    :, 0, :
].T  # (n_samples, obs_dim)
interleaved = torch.arange(0, 2000, step=2)
x_train_convolved, x_test_convolved = (
    x_convolved[interleaved],
    x_convolved[interleaved + 1],
)
z_train, z_test = (
    z[interleaved],
    z[interleaved + 1],
)  # (n_samples, n_components)
x_train, x_test = (
    x[interleaved],
    x[interleaved + 1],
)  # (n_samples, obs_dim)
x_input_train, x_input_test = (
    x_input[interleaved],
    x_input[interleaved + 1],
)  # (n_samples, obs_dim)
train_dataloader = DataLoader(
    TensorDataset(x_train, x_train_convolved, z_train, x_input_train),
    batch_size=128,
    shuffle=True,
)
n_components, obs_dim = z_train.shape[1], x_train.shape[1]


tag = "baseline"
results_file = f"results_{tag}"


## initialization
torch.manual_seed(seed)

if method in ["lrRNN", "lrRNN+ICA", "LINT"]:
    n_groups = 1
    alpha = 0
    beta = 0
elif method == "bdRNN":
    n_groups = n_groups_true
    alpha = 1
    beta = 0
elif method == "DisRNN-full":
    n_groups = n_components
    alpha = 0
    beta = 20
elif method == "DisRNN":
    n_groups = n_groups_true
    alpha = 0
    beta = 20
else:
    raise ValueError(f"{method} is not a valid method")

n_total_samples = x_train.shape[0]
vae = utils.VAE(
    obs_dim=obs_dim, n_components=n_components, n_total_samples=n_total_samples
)


## Lightning module
n_epochs = 5000
learning_rate = 1e-3

lit_vae = utils.LitVAE(vae, n_groups, alpha, beta, n_total_samples, learning_rate)

if args.mode in ["train", "both"]:
    wandb_logger = WandbLogger(
        name=name,
        project=f"iclr2026-{__file__.split('/')[-2]}",
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
        model=lit_vae,
        train_dataloaders=train_dataloader,
    )
    trainer.save_checkpoint(f"{results_file}/{name}.ckpt")

## evaluation
if args.mode in ["eval", "both"]:
    if method in ["lrRNN+ICA", "LINT"]:
        lit_vae = utils.LitVAE.load_from_checkpoint(
            f"{results_file}/lrRNN_{seed}.ckpt", vae=vae
        )
    else:
        lit_vae = utils.LitVAE.load_from_checkpoint(
            f"{results_file}/{name}.ckpt", vae=vae
        )

    n_monte_carlo = 10
    df_result = pd.DataFrame(
        index=np.arange(1),
        columns=[
            "conditional log-likelihood",
            "reconstruction $R^2$",
            "partial correlation",
            "total correlation",
            "group latent $R^2$",
            "component latent $R^2$",
            "connectivity error",
            "connectivity RMSE",
            "connectivity correlation",
            "connectivity $R^2$",
            "sub-connectivity error",
            "sub-connectivity RMSE",
            "sub-connectivity correlation",
            "sub-connectivity $R^2$",
        ],
    )
    with torch.no_grad():
        z_pred_mean, z_pred_log_std = vae.encode(
            x_test_convolved
        )  # (n_samples, n_components)
        x_pred_mean, x_pred_log_std = vae.decode(z_pred_mean, x_input_test)

        df_result.at[0, "conditional log-likelihood"] = (
            -F.gaussian_nll_loss(
                x_pred_mean,
                x_test,
                x_pred_log_std.exp() ** 2,
                full=True,
                reduction="none",
            )
            .sum(dim=-1)
            .mean()
            .item()
        )
        df_result.at[0, "reconstruction $R^2$"] = r2_score(x_pred_mean, x_test).item()
        if method == "lrRNN+ICA":
            ica = FastICA(n_components=n_components, random_state=0)
            ica.fit(z_pred_mean.numpy())
            z_pred_mean = torch.from_numpy(ica.transform(z_pred_mean.numpy())).float()
            A = vae.decoder.linear.weight @ torch.from_numpy(ica.mixing_).float()
            B = torch.from_numpy(ica.components_).float() @ vae.encoder.linear.weight
        elif method == "LINT":
            A, S, B = torch.linalg.svd(
                vae.decoder.linear.weight @ vae.encoder.linear.weight
            )
            A, B = A[:, :n_components] * S[:n_components], B[:n_components, :]
            z_pred_mean = x_test_convolved @ B.T
        else:
            A, B = vae.decoder.linear.weight, vae.encoder.linear.weight
        torch.save(A, f"{results_file}/{name}_A.pt")
        torch.save(B, f"{results_file}/{name}_B.pt")

        df_result.at[0, "partial correlation"] = (
            disentangle.estimate_partial_correlation(
                z_pred_mean,
                z_pred_log_std,
                n_groups=n_groups_true,
                n_monte_carlo=n_monte_carlo,
            ).item()
        )
        df_result.at[0, "total correlation"] = disentangle.estimate_partial_correlation(
            z_pred_mean,
            z_pred_log_std,
            n_groups=n_components,
            n_monte_carlo=n_monte_carlo,
        ).item()

        group_aligned_z_pred_mean = disentangle.align(
            z_pred_mean,
            z_test,
            n_groups_true=n_groups_true,
            n_groups_pred=n_groups_true,
        )

        df_result.at[0, "group latent $R^2$"] = r2_score(
            group_aligned_z_pred_mean, z_test
        ).item()

        if method not in ["bdRNN", "DisRNN"]:
            all_r2_scores = disentangle.permute_align(
                z_pred_mean,
                z_test,
                n_groups_true,
            )
            torch.save(
                all_r2_scores,
                f"{results_file}/{name}_all_r2_scores.pt",
            )

        component_aligned_z_pred_mean = disentangle.align(
            z_pred_mean, z_test, n_groups_true=n_components, n_groups_pred=n_components
        )
        df_result.at[0, "component latent $R^2$"] = r2_score(
            component_aligned_z_pred_mean, z_test
        ).item()

        z_pred_mean, z_pred_log_std = vae.encode(
            x_convolved
        )  # (n_samples, n_components)
        if method == "lrRNN+ICA":
            ica = FastICA(n_components=n_components, random_state=0)
            ica.fit(z_pred_mean.numpy())
            z_pred_mean = torch.from_numpy(ica.transform(z_pred_mean.numpy())).float()
        elif method == "LINT":
            _, _, BB = torch.linalg.svd(
                vae.decoder.linear.weight @ vae.encoder.linear.weight
            )
            BB = BB[:n_components, :]
            z_pred_mean = x_test_convolved @ BB.T
        group_aligned_z_pred_mean = disentangle.align(
            z_pred_mean, z, n_groups_true=n_groups_true, n_groups_pred=n_groups_true
        )
        torch.save(
            group_aligned_z_pred_mean,
            f"{results_file}/{name}_group_aligned_z_pred_mean.pt",
        )

        W_true = (df_data.at[0, "A"] @ df_data.at[0, "B"]).flatten()
        W = (A @ B).flatten()
        df_result.at[0, "connectivity error"] = (((W - W_true).abs()).mean()).item()
        df_result.at[0, "connectivity RMSE"] = (
            ((W - W_true) ** 2).mean() ** 0.5
        ).item()
        df_result.at[0, "connectivity $R^2$"] = r2_score(W, W_true).item()
        df_result.at[0, "connectivity correlation"] = torch.corrcoef(
            torch.stack(
                [
                    W,
                    W_true,
                ],
                dim=0,
            )
        )[0, 1].item()

        W_sub_true = torch.stack(
            [
                (df_data.at[0, "A"][:, :3] @ df_data.at[0, "B"][:3, :]).flatten(),
                (df_data.at[0, "A"][:, 3:] @ df_data.at[0, "B"][3:, :]).flatten(),
            ],
            dim=1,
        )
        W_sub = torch.stack(
            [
                (A[:, :3] @ B[:3, :]).flatten(),
                (A[:, 3:] @ B[3:, :]).flatten(),
            ],
            dim=1,
        )
        if (W_sub[:, 0] - W_sub_true[:, 0]).abs().mean() + (
            W_sub[:, 1] - W_sub_true[:, 1]
        ).abs().mean() > (W_sub[:, 0] - W_sub_true[:, 1]).abs().mean() + (
            W_sub[:, 1] - W_sub_true[:, 0]
        ).abs().mean():
            W_sub = torch.stack([W_sub[:, 1], W_sub[:, 0]], dim=1)

        df_result.at[0, "sub-connectivity error"] = (
            ((W_sub - W_sub_true).abs()).mean()
        ).item()
        df_result.at[0, "sub-connectivity RMSE"] = (
            ((W_sub - W_sub_true) ** 2).mean() ** 0.5
        ).item()
        df_result.at[0, "sub-connectivity $R^2$"] = r2_score(W_sub, W_sub_true).item()
        df_result.at[0, "sub-connectivity correlation"] = torch.corrcoef(
            torch.stack(
                [
                    W_sub.flatten(),
                    W_sub_true.flatten(),
                ],
                dim=0,
            )
        )[0, 1].item()

    df_result.to_csv(f"{results_file}/{name}.csv", index=False)
