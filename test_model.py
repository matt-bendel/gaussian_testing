import numpy as np
import torch
import yaml
import types
import json

import matplotlib.pyplot as plt

from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.rcGAN import rcGAN
from models.lightning.PCANET import PCANET
from data.lightning.GaussianDataModule import DataTransform


class Posterior:
    def __init__(self, d, mu_x, Sig_xx, sig_noise, y):
        inds = [0, 1, 5, 8, 9]

        A = np.eye(d)
        A[inds, :] = 0

        self.mu_x = np.expand_dims(mu_x, axis=1)
        self.y = np.expand_dims(y, axis=1)
        self.Sig_xx = Sig_xx
        self.mu_y = A @ self.mu_x
        self.Sig_yy = A @ Sig_xx @ A.T + sig_noise * np.eye(d)
        self.Sig_yy_inv = np.linalg.inv(self.Sig_yy)
        self.Sig_xy = Sig_xx @ A.T
        self.Sig_yx = A @ Sig_xx

        self.posterior_mean = self.mu_x + self.Sig_xy @ self.Sig_yy_inv @ (self.y - self.mu_y)
        self.posterior_cov = self.Sig_xx - self.Sig_xy @ self.Sig_yy_inv @ self.Sig_yx


def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)

    mu_x = np.load('/Users/mattbendel/Documents/pca_reg_simple/data/stats/gt_mu.npy')
    e_vals = np.abs(np.load('/Users/mattbendel/Documents/pca_reg_simple/data/stats/gt_e_vals.npy'))
    e_vecs = np.load('/Users/mattbendel/Documents/pca_reg_simple/data/stats/gt_e_vecs.npy')

    cov_x = e_vecs.T @ np.diag(e_vals) @ e_vecs
    sig_noise = 0.001

    dt = DataTransform()

    avg_cov = np.zeros((10, 10))
    for i in range(10000):
        x = torch.from_numpy(np.random.multivariate_normal(mu_x, cov_x, 1))
        x, y, mask = dt(x)

        posterior = Posterior(10, mu_x, cov_x, sig_noise, y[0].numpy())

        x_z_m = x.numpy().T - posterior.posterior_mean

        avg_cov += x_z_m @ x_z_m.T

    avg_cov = avg_cov / 10000

    num_z = 1000

