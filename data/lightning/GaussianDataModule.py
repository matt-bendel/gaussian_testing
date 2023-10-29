import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import scipy as sp

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

    def cov_dist(self, posterior_cov_hat):
        e_vals = np.real(sp.linalg.eigvals(self.posterior_cov, posterior_cov_hat))
        rho = np.sqrt(np.sum(np.log(e_vals) ** 2))

        return rho

    def cfid(self):
        pass

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            return self.transform(x)

        return x

    def __len__(self):
        return len(self.data)

class DataTransform:
    def __init__(self, d):
        self.M = d
        self.A = torch.eye(self.M)
        self.mask = torch.ones(d)

        if d == 10:
            self.A[[0, 1, 5, 8, 9], :] = 0
            self.mask[[0, 1, 5, 8, 9]] = 0
        else:
            self.A[np.arange(d // 2) * 2, :] = 0
            self.mask[np.arange(d // 2) * 2] = 0

        self.w_mu = torch.zeros(self.M)
        self.w_cov = torch.eye(self.M) * 0.001

        mu = np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{d}d/gt_mu.npy')
        e_vals = np.abs(np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{d}d/gt_e_vals.npy'))
        e_vecs = np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{d}d/gt_e_vecs.npy')

        self.mu = mu
        self.cov = e_vecs @ np.diag(e_vals) @ e_vecs.T

    def __call__(self, x):
        y = x * self.mask + torch.from_numpy(np.random.multivariate_normal(self.w_mu, self.w_cov)) + 0.0 * torch.ones_like(x)

        x = x.to(torch.float32)
        y = y.to(torch.float32)
        mask = self.mask.to(torch.float32)

        # posterior = Posterior(10, self.mu, self.cov, 0.001, y.numpy())

        # e_vals, e_vecs = np.linalg.eigh(posterior.posterior_cov)

        return x, y, mask #, torch.from_numpy(e_vals).to(torch.float32), torch.from_numpy(e_vecs).to(torch.float32)


class GaussianDataModule(pl.LightningDataModule):
    def __init__(self, d):
        super().__init__()
        self.batch_size = 32
        self.num_workers = 8
        mu = np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{d}d/gt_mu.npy')
        e_vals = np.abs(np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{d}d/gt_e_vals.npy'))
        e_vecs = np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{d}d/gt_e_vecs.npy')

        self.d = d
        self.mu = mu
        self.cov = e_vecs @ np.diag(e_vals) @ e_vecs.T

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        transform = DataTransform(self.d)
        data = CustomTensorDataset(
            torch.from_numpy(np.random.multivariate_normal(self.mu, self.cov, 100000)),
            transform=transform
        )
        self.train_data, self.val_data, self.test_data = random_split(data, [70000, 20000, 10000])

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)
