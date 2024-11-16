import numpy as np
import torch
from scipy import linalg


def cal_fid(pred_motion: torch.Tensor, gt_motions: torch.Tensor, eps=1e-6):
    gt_mu, gt_cov = calculate_activation_statistics(gt_motions.float())
    pred_mu, pred_cov = calculate_activation_statistics(pred_motion.float())
    fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov, eps=eps)
    return fid


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculates the Frechet Distance between two multivariate Gaussians.

    The Frechet distance between two multivariate Gaussians X1 ~ N(mu1, sigma1)
    and X2 ~ N(mu2, sigma2) is:

        d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 * sigma2))

    Stable version by Dougal J. Sutherland.

    :param mu1: PyTorch tensor containing the mean of activations for generated samples.
    :param sigma1: PyTorch tensor containing the covariance matrix over activations for generated samples.
    :param mu2: PyTorch tensor containing the mean of activations for real samples.
    :param sigma2: PyTorch tensor containing the covariance matrix over activations for real samples.
    :param eps: Small value added to the diagonal of covariance matrices for numerical stability.
    :return: The Frechet Distance.
    """
    if isinstance(mu1, torch.Tensor):
        mu1 = mu1.detach().float().cpu().numpy()
        mu2 = mu2.detach().float().cpu().numpy()
        sigma1 = sigma1.detach().float().cpu().numpy()
        sigma2 = sigma2.detach().float().cpu().numpy()

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(activations: torch.Tensor):
    """
    Calculates the mean and covariance of activations.

    :param activations: PyTorch tensor of shape (batch_size, features).
    :return: Tuple containing the mean vector and covariance matrix.
    """
    mu = torch.mean(activations, dim=0)
    activations_minus_mu = activations - mu
    cov = (activations_minus_mu.T @ activations_minus_mu) / (activations.shape[0] - 1)
    return mu, cov

