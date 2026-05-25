import itertools

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torcheval.metrics.functional import r2_score


def kl_div_normal(
    z_pred_mean: torch.Tensor, z_pred_log_std: torch.Tensor
) -> torch.Tensor:
    """Analytical KL divergence between a normal distribution and the normal prior distribution.

    KL(q||p) = 0.5 * (tr(Σ_p^-1 Σ_q) + (μ_p - μ_q)^T Σ_p^-1 (μ_p - μ_q) - k + log(|Σ_p|/|Σ_q|))

    When p is a standard normal distribution, the KL divergence simplifies to:
    KL(q||p) = 0.5 * (μ^T μ + tr(Σ) - k - log|Σ|)

    Here we also assume Σ is diagonal, so the KL divergence simplifies to:
    KL(q||p) = 0.5 * (μ^T μ + σ^T σ - k - 2 * sum(log(σ)))

    https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/

    Parameters
    ----------
    z_pred_mean : torch.Tensor of shape (*, n_components)
        The predicted mean of the latent variable.
    z_pred_log_std : torch.Tensor of shape (*, n_components)
        The predicted log standard deviation of the latent variable.

    Returns
    -------
    torch.Tensor of shape (*,)
        The analytical KL divergence.
    """
    return (
        0.5 * (z_pred_mean**2 + z_pred_log_std.exp() ** 2 - 1 - 2 * z_pred_log_std)
    ).sum(dim=-1)


def aggregated_posterior(
    z_pred_mean: torch.Tensor,
    z: torch.Tensor,
    z_pred_log_std: torch.Tensor,
    n_groups: int,
    n_total_samples: int = 1e7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregated posterior jointly and dimension-wise.

    Parameters
    ----------
    z_pred_mean : torch.Tensor of shape (batch_size, n_components)
        The predicted mean of the latent variable.
    z : torch.Tensor of shape (batch_size, n_components)
        The sampled latent variable.
    z_pred_log_std : torch.Tensor of shape (batch_size, n_components)
        The predicted log standard deviation of the latent variable.

    Returns
    -------
    ln_q_z : torch.Tensor of shape (batch_size,)
        The joint aggregated posterior.
    ln_prod_q_zi : torch.Tensor of shape (batch_size,)
        The dimension-wise aggregated posterior.
    """
    batch_size, n_components = z_pred_mean.shape
    group_rank = n_components // n_groups
    n_monte_carlo = batch_size
    mat_ln_q_z = -F.gaussian_nll_loss(
        z_pred_mean.view((1, batch_size, n_groups, group_rank)),
        z.view((n_monte_carlo, 1, n_groups, group_rank)),
        (z_pred_log_std.exp() ** 2).view((1, batch_size, n_groups, group_rank)),
        full=True,
        reduction="none",
    )  # (n_monte_carlo = batch_size, batch_size, n_groups, group_rank)

    reweights = (
        torch.ones(batch_size, batch_size, device=z.device)
        / (batch_size - 1)
        * (n_total_samples - 1)
    )
    reweights[torch.arange(batch_size), torch.arange(batch_size)] = 1
    reweights = reweights.log()

    ln_q_z = torch.logsumexp(mat_ln_q_z.sum(dim=(2, 3)) + reweights, dim=1) - np.log(
        n_total_samples
    )
    ln_prod_q_zi = (
        torch.logsumexp(mat_ln_q_z.sum(dim=3) + reweights.unsqueeze(-1), dim=1)
        - np.log(n_total_samples)
    ).sum(dim=1)
    return ln_q_z, ln_prod_q_zi


@torch.no_grad()
def estimate_partial_correlation(
    z_pred_mean: torch.Tensor,
    z_pred_log_std: torch.Tensor,
    n_groups: int,
    n_monte_carlo: int = 1,
    seed: int = 0,
) -> torch.Tensor:
    """Estimate the partial correlation of the latent variable.

    Parameters
    ----------
    z_pred_mean : torch.Tensor of shape (batch_size, n_components)
        The predicted mean of the latent variable.
    z_pred_log_std : torch.Tensor of shape (batch_size, n_components)
        The predicted log standard deviation of the latent variable.
    n_groups : int
        The number of groups.
    n_monte_carlo : int, optional
        The number of Monte Carlo samples, by default 1.
    seed : int, optional
        Random seed, by default 0.

    Returns
    -------
    torch.Tensor
        The estimated partial correlation.
    """
    n_points, n_components = z_pred_mean.shape

    if n_components % n_groups != 0:
        raise ValueError("group_rank = n_components / n_groups is not an integer.")
    group_rank = int(n_components / n_groups)

    generator = torch.Generator().manual_seed(seed)

    z_sampled = (
        torch.randn((n_monte_carlo, n_points, n_components), generator=generator)
        * z_pred_log_std.exp()
        + z_pred_mean
    )
    mat_ln_q_z = -F.gaussian_nll_loss(
        z_pred_mean.view((1, n_points, n_groups, group_rank)),
        z_sampled.view((n_monte_carlo * n_points, 1, n_groups, group_rank)),
        (z_pred_log_std.exp() ** 2).view((1, n_points, n_groups, group_rank)),
        full=True,
        reduction="none",
    )  # (n_monte_carlo*n_points, n_points, n_groups, group_rank)

    ln_q_z = torch.logsumexp(mat_ln_q_z.sum(dim=(2, 3)), dim=1) - np.log(n_points)
    ln_prod_q_zi = (
        torch.logsumexp(mat_ln_q_z.sum(dim=3), dim=1) - np.log(n_points)
    ).sum(dim=1)
    return (ln_q_z - ln_prod_q_zi).mean()


@torch.no_grad()
def align(
    z_pred: torch.Tensor, z_true: torch.Tensor, n_groups_true: int, n_groups_pred: int
) -> torch.Tensor:
    """Align the predicted latent variable with the true latent variable.

    Parameters
    ----------
    z_pred : torch.Tensor of shape (n_samples, n_pred_components)
        The predicted latent variable.
    z_true : torch.Tensor of shape (n_samples, n_true_components)
        The true latent variable.
    n_groups : int, optional
        Number of groups, by default 1.

    Returns
    -------
    aligned_z_pred : torch.Tensor of shape (n_samples, n_components)
        The aligned predicted latent variable.
    """
    n_samples, n_components_true = z_true.shape
    _, n_components_pred = z_pred.shape
    if n_groups_pred < n_groups_true:
        raise ValueError(
            "The number of groups in the predicted latent variable must be greater than or equal to the number of groups in the true latent variable."
        )
    group_rank_true = int(n_components_true / n_groups_true)
    group_rank_pred = int(n_components_pred / n_groups_pred)

    aligned_z_all = torch.zeros(
        (n_groups_true, n_groups_pred, n_samples, group_rank_true)
    )
    r2_matrix = torch.zeros((n_groups_true, n_groups_pred))
    for true_group in range(n_groups_true):
        for pred_group in range(n_groups_pred):
            z_group_true = z_true.view(n_samples, n_groups_true, -1)[:, true_group, :]
            z_group_pred = z_pred.view(n_samples, n_groups_pred, -1)[:, pred_group, :]
            z_aug = torch.cat([torch.ones(n_samples, 1), z_group_pred], dim=1)
            wtsaffine = torch.linalg.lstsq(z_aug, z_group_true).solution
            aligned_z_all[true_group, pred_group] = z_aug @ wtsaffine
            r2_matrix[true_group, pred_group] = r2_score(
                aligned_z_all[true_group, pred_group], z_group_true
            )

    row_ind, col_ind = linear_sum_assignment(r2_matrix, maximize=True)
    print(r2_matrix)
    print(row_ind, col_ind)
    aligned_z_pred = (
        aligned_z_all[row_ind, col_ind]
        .transpose(0, 1)
        .reshape(n_samples, n_components_true)
    )  # (n_samples, n_components_true)
    return aligned_z_pred


@torch.no_grad()
def permute_align(
    z_pred: torch.Tensor, z_true: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Align the predicted latent variable with the true latent variable, by all possible permutations.

    Parameters
    ----------
    z_pred : torch.Tensor of shape (n_samples, n_pred_components)
        The predicted latent variable.
    z_true : torch.Tensor of shape (n_samples, n_true_components)
        The true latent variable.
    n_groups : int, optional
        Number of groups, by default 1.

    Returns
    -------
    aligned_z_pred : torch.Tensor of shape (n_samples, n_components)
        The aligned predicted latent variable.
    """
    n_samples, n_components = z_true.shape
    _, n_components = z_pred.shape
    group_rank = int(n_components / n_groups)

    all_permutations = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 3, 2, 4, 5],
        [0, 1, 4, 2, 3, 5],
        [0, 1, 5, 2, 3, 4],
        [0, 2, 3, 1, 4, 5],
        [0, 2, 4, 1, 3, 5],
        [0, 2, 5, 1, 3, 4],
        [0, 3, 4, 1, 2, 5],
        [0, 3, 5, 1, 2, 4],
        [0, 4, 5, 1, 2, 3],
    ]

    all_r2_scores = []

    best_score = -float("inf")
    best_aligned = None

    for perm in all_permutations:
        z_perm = z_pred[:, perm]

        aligned_z_all = torch.zeros((n_groups, n_groups, n_samples, group_rank))
        r2_matrix = torch.zeros((n_groups, n_groups))
        for true_group in range(n_groups):
            for pred_group in range(n_groups):
                z_group_true = z_true.view(n_samples, n_groups, -1)[:, true_group, :]
                z_group_pred = z_perm.view(n_samples, n_groups, -1)[:, pred_group, :]
                z_aug = torch.cat([torch.ones(n_samples, 1), z_group_pred], dim=1)
                wtsaffine = torch.linalg.lstsq(z_aug, z_group_true).solution
                aligned_z_all[true_group, pred_group] = z_aug @ wtsaffine
                r2_matrix[true_group, pred_group] = r2_score(
                    aligned_z_all[true_group, pred_group], z_group_true
                )

        row_ind, col_ind = linear_sum_assignment(r2_matrix, maximize=True)
        # aligned_z_pred = (
        #     aligned_z_all[row_ind, col_ind]
        #     .transpose(0, 1)
        #     .reshape(n_samples, n_components)
        # )

        score = r2_matrix[row_ind, col_ind].mean().item()
        all_r2_scores.append(score)

    return all_r2_scores
