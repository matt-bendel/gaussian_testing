import numpy
import numpy as np
import torch
import yaml
import types
import json

import matplotlib.pyplot as plt
import scipy as sp

from data.lightning.GaussianDataModule import DataTransform
from data.lightning.GaussianDataModule import GaussianDataModule
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

if __name__ == '__main__':
    for d in range(10):
        d = (d + 1) * 10
        dm = GaussianDataModule(d)
        dm.setup()

        mu_x = np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{d}d/gt_mu.npy')
        e_vals = np.abs(np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{d}d/gt_e_vals.npy'))
        e_vecs = np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{d}d/gt_e_vecs.npy')

        cov_x = e_vecs @ np.diag(e_vals) @ e_vecs.T
        sig_noise = 0.001
        dt = DataTransform(d)

        x = torch.from_numpy(np.random.multivariate_normal(mu_x, cov_x, 1000))
        x, y, mask = dt(x)

        posterior = Posterior(d, mu_x, cov_x, sig_noise, y[-1].numpy())

        train_loader = dm.train_dataloader()

        mu_x = []
        mu_y = []
        c_xx = []
        c_yy = []
        c_xy = []
        c_yx = []

        with torch.no_grad():
            for i, data in enumerate(train_loader):
                x, y, mask = data

                mu_x.append(torch.mean(x, dim=0))
                mu_y.append(torch.mean(y, dim=0))

                B, D = x.size()
                mean_x = x.mean(dim=0)
                mean_y = y.mean(dim=0)

                diffs_x = (x - mean_x[None, :])
                diffs_y = (y - mean_y[None, :])

                prods = torch.bmm(diffs_x.unsqueeze(2), diffs_x.unsqueeze(1)).reshape(B, D, D)
                c_xx.append(prods.sum(dim=0) / (B - 1))  # Unbiased estimate

                prods = torch.bmm(diffs_y.unsqueeze(2), diffs_y.unsqueeze(1)).reshape(B, D, D)
                c_yy.append(prods.sum(dim=0) / (B - 1))  # Unbiased estimate

                prods = torch.bmm(diffs_x.unsqueeze(2), diffs_y.unsqueeze(1)).reshape(B, D, D)
                c_xy.append(prods.sum(dim=0) / (B - 1))  # Unbiased estimate

                prods = torch.bmm(diffs_y.unsqueeze(2), diffs_x.unsqueeze(1)).reshape(B, D, D)
                c_yx.append(prods.sum(dim=0) / (B - 1))  # Unbiased estimate

            mu_x = torch.mean(torch.stack(mu_x, dim=0), dim=0).cpu().numpy()
            mu_y = torch.mean(torch.stack(mu_y, dim=0), dim=0).cpu().numpy()

            c_xx = torch.mean(torch.stack(c_xx, dim=0), dim=0).cpu().numpy()
            c_yy = torch.mean(torch.stack(c_yy, dim=0), dim=0).cpu().numpy()
            c_xy = torch.mean(torch.stack(c_xy, dim=0), dim=0).cpu().numpy()
            c_yx = torch.mean(torch.stack(c_yx, dim=0), dim=0).cpu().numpy()

            posterior_cov_hat_final = c_xx - c_xy @ np.linalg.inv(c_yy) @ c_yx



            # self.posterior_cov = self.Sig_xx - self.Sig_xy @ self.Sig_yy_inv @ self.Sig_yx
            # e_vals, e_vecs = np.linalg.eigh(posterior.posterior_cov)
            #
            # Z_weights = e_vecs @ np.diag(np.sqrt(e_vals)) @ e_vecs.T
            # Z_bias = np.zeros(d)
            #
            # Y_weights = posterior.Y_weight_mat
            # Y_bias = posterior.Y_bias
            #
            # Z_weights = torch.from_numpy(Z_weights)
            # Z_bias = torch.from_numpy(Z_bias)
            #
            # Y_weights = torch.from_numpy(Y_weights)
            # Y_bias = torch.from_numpy(Y_bias)
            #
            # lin_y = nn.Linear(d, d)
            # lin_z = nn.Linear(d, d, bias=False)

            # lin_y.weight.copy_(Y_weights)
            # lin_y.bias.copy_(Y_bias[:, 0])
            #
            # lin_z.weight.copy_(Z_weights)
            #
            # posterior_cov_hat_final = numpy.zeros((d, d))
            #
            # for i in range(y.shape[0]):
            #     z = torch.randn(1000, d)
            #
            #     out1 = lin_y(torch.unsqueeze(y[i, :], dim=0))
            #     out2 = lin_z(z)
            #
            #     x_hat = out1 + out2
            #     x_hat = x_hat.numpy()
            #
            #     posterior_mean_hat = np.mean(x_hat, axis=0)
            #
            #     gens_zero_mean = x_hat - posterior_mean_hat[None, :]
            #
            #     posterior_cov_hat = 1 / (x_hat.shape[0] - 1) * gens_zero_mean.T @ gens_zero_mean
            #     posterior_cov_hat_final += posterior_cov_hat
            #
            # posterior_cov_hat_final = 1 / y.shape[0] * posterior_cov_hat_final

            print(f'CFID for d={d}: {posterior.cfid(posterior_cov_hat_final)}')

