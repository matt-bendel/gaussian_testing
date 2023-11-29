import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
import torch

# start w/ d= 100
# decrease d
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
e_vecs, _ = qr(np.random.randn(d, d))
mu = np.random.randn(d)

c_xx = e_vecs @ np.diag(e_vals) @ e_vecs.T
mu_xx = mu

for i in range(10):
    dim = 10 - i
    new_mu = mu_xx[0:dim*10]

    new_evecs_x = e_vecs[:, 0:dim*10]
    new_evecs_x = new_evecs_x[0:dim*10, :]
    new_e_vecs, _ = qr(new_evecs_x)
    new_e_vals = e_vals[0:dim*10]

    print(new_e_vecs.shape)
    print(new_e_vals.shape)

    # temp = input('Save?')
    #
    # if temp == '1':
    np.save(f'data/stats_{dim*10}d/gt_e_vals.npy', new_e_vals)
    np.save(f'data/stats_{dim*10}d/gt_e_vecs.npy', new_e_vecs)
    np.save(f'data/stats_{dim*10}d/gt_mu.npy', new_mu)
