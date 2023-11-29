import numpy
import numpy as np
import torch
import yaml
import types
import json

import matplotlib.pyplot as plt
import scipy as sp

from data.lightning.GaussianDataModule import DataTransform
from torch import nn

class Posterior:
    def __init__(self, d, mu_x, Sig_xx, sig_noise, y):
        inds = np.arange(d // 2) * 2

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
        self.Y_weight_mat = self.Sig_xy @ self.Sig_yy_inv
        self.Y_bias = self.mu_x - self.Y_weight_mat @ self.mu_y

        self.posterior_mean = self.mu_x + self.Sig_xy @ self.Sig_yy_inv @ (self.y - self.mu_y)
        self.posterior_cov = self.Sig_xx - self.Sig_xy @ self.Sig_yy_inv @ self.Sig_yx
        self.posterior_cov_sqrt = sp.linalg.sqrtm(self.posterior_cov)

    def get_new_mean(self, y):
        return self.mu_x + self.Sig_xy @ self.Sig_yy_inv @ (np.expand_dims(y, axis=1) - self.mu_y)

    def cfid(self, posterior_cov_hat):
        # mu_dist = np.linalg.norm(self.posterior_mean - posterior_mean_hat) ** 2
        cov_sum = posterior_cov_hat + self.posterior_cov
        posterior_prod_sqrt = sp.linalg.sqrtm(self.posterior_cov_sqrt @ posterior_cov_hat @ self.posterior_cov_sqrt)

        return np.trace(cov_sum - 2 * posterior_prod_sqrt)



mu_x = np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_100d/gt_mu.npy')
e_vals = np.abs(np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_100d/gt_e_vals.npy'))
e_vecs = np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_100d/gt_e_vecs.npy')

cov_x = e_vecs @ np.diag(e_vals) @ e_vecs.T
sig_noise = 0.001
dt = DataTransform(100)

x = torch.from_numpy(np.random.multivariate_normal(mu_x, cov_x, 1000))
x, y, mask = dt(x)

posterior = Posterior(100, mu_x, cov_x, sig_noise, y[-1].numpy())
e_vals, e_vecs = np.linalg.eigh(posterior.posterior_cov)

Z_weights = e_vecs @ np.diag(np.sqrt(e_vals)) @ e_vecs.T
Z_bias = np.zeros(100)

Y_weights = posterior.Y_weight_mat
Y_bias = posterior.Y_bias

Z_weights = torch.from_numpy(Z_weights)
Z_bias = torch.from_numpy(Z_bias)

Y_weights = torch.from_numpy(Y_weights)
Y_bias = torch.from_numpy(Y_bias)

lin_y = nn.Linear(100, 100)
lin_z = nn.Linear(100, 100, bias=False)

with torch.no_grad():
    lin_y.weight.copy_(Y_weights)
    lin_y.bias.copy_(Y_bias[:, 0])

    lin_z.weight.copy_(Z_weights)

    posterior_cov_hat_final = numpy.zeros((100, 100))

    for i in range(y.shape[0]):
        z = torch.randn(1000, 100)

        out1 = lin_y(torch.unsqueeze(y[i, :], dim=0))
        out2 = lin_z(z)

        x_hat = out1 + out2
        x_hat = x_hat.numpy()

        posterior_mean_hat = np.mean(x_hat, axis=0)

        gens_zero_mean = x_hat - posterior_mean_hat[None, :]

        posterior_cov_hat = 1 / (x_hat.shape[0] - 1) * gens_zero_mean.T @ gens_zero_mean
        posterior_cov_hat_final += posterior_cov_hat

    posterior_cov_hat_final = 1 / y.shape[0] * posterior_cov_hat_final

    print(f'CFID: {posterior.cfid(posterior_cov_hat_final)}')

