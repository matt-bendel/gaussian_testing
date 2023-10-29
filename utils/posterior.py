import numpy as np
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
        self.posterior_cov_sqrt = sp.linalg.sqrtm(self.posterior_cov)

    def cov_dist(self, posterior_cov_hat):
        e_vals = np.real(sp.linalg.eigvals(self.posterior_cov, posterior_cov_hat))
        rho = np.sqrt(np.sum(np.log(e_vals) ** 2))

        return rho