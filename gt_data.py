import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import torch

class Posterior:
    def __init__(self, d, mu_x, Sig_xx, sig_noise, y):
        inds = np.arange(d//2) * 2

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

d = 100
inds = np.arange(d//2) * 2
print(inds)
A = np.eye(d)
A[inds, :] = 0

e_vals = np.abs(np.random.randn(d)) / 2

plt.scatter(np.arange(d), e_vals)
plt.savefig('prior_evals.png')
e_vecs = np.random.randn(d, d)
mu = np.random.randn(d)

c_xx = e_vecs @ np.diag(e_vals) @ e_vecs.T
mu_xx = mu

x = np.random.multivariate_normal(mu_xx, c_xx)
y = A @ x + np.random.multivariate_normal(np.zeros(d), np.eye(d) * 0.001)

posterior = Posterior(d, mu_xx, c_xx, 0.001, y)

e_vals_post, _ = np.linalg.eigh(posterior.posterior_cov)

plt.scatter(np.arange(d), e_vals_post)
plt.savefig('posterior_evals.png')

temp = input('Save?')

if temp == '1':
    np.save(f'data/stats_{d}d/gt_e_vals.npy', e_vals)
    np.save(f'data/stats_{d}d/gt_e_vecs.npy', e_vecs)
    np.save(f'data/stats_{d}d/gt_mu.npy', mu_xx)
