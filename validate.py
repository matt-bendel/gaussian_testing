import numpy
import numpy as np
import torch
import yaml
import types
import json

# TODO: go back to harder model
import matplotlib.pyplot as plt
import scipy as sp

from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.rcGAN import rcGAN
from models.lightning.rcGAN_w_pca_reg import rcGANwPCAReg
from models.lightning.rcGAN_w_lazy_reg import rcGANwLazyReg
from models.lightning.rcGAN_w_lazy_reg_simple_gen import rcGANwLazyRegSimple
from models.lightning.rcGAN_no_std_dev import rcGANNoStdDev
from models.lightning.PCANET import PCANET
from data.lightning.GaussianDataModule import DataTransform

# TODO: Compute
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

        self.posterior_mean = self.mu_x + self.Sig_xy @ self.Sig_yy_inv @ (self.y - self.mu_y)
        self.posterior_cov = self.Sig_xx - self.Sig_xy @ self.Sig_yy_inv @ self.Sig_yx
        self.posterior_cov_sqrt = sp.linalg.sqrtm(self.posterior_cov)

    def cov_dist(self, posterior_cov_hat):
        e_vals = np.real(sp.linalg.eigvals(self.posterior_cov, posterior_cov_hat))
        rho = np.sqrt(np.sum(np.log(e_vals) ** 2))

        return rho

    def cfid(self, posterior_cov_hat):
        # mu_dist = np.linalg.norm(self.posterior_mean - posterior_mean_hat) ** 2
        cov_sum = posterior_cov_hat + self.posterior_cov
        posterior_prod_sqrt = sp.linalg.sqrtm(self.posterior_cov_sqrt @ posterior_cov_hat @ self.posterior_cov_sqrt)

        return np.trace(cov_sum - 2 * posterior_prod_sqrt)


def load_object(dct):
    return types.SimpleNamespace(**dct)


def compute_posterior_stats(generator, y, mask, d):
    num_z = d*10

    gens = np.zeros((num_z, d))
    for z in range(num_z):
        with torch.no_grad():
            gens[z, :] = generator.forward(y.unsqueeze(0), mask.unsqueeze(0))[0, 0, :].numpy()

    posterior_mean_hat = np.mean(gens, axis=0)

    gens_zero_mean = gens - posterior_mean_hat[None, :]

    posterior_cov_hat = 1 / (gens.shape[0] - 1) * gens_zero_mean.T @ gens_zero_mean

    return posterior_mean_hat, posterior_cov_hat


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")

    with open('configs/rcgan.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    # Server
    mu_x = np.load(f'/home/bendel.8/Git_Repos/gaussian_testing/data/stats_{args.d}d/gt_mu.npy')
    e_vals = np.abs(np.load(f'/home/bendel.8/Git_Repos/gaussian_testing/data/stats_{args.d}d/gt_e_vals.npy'))
    e_vecs = np.load(f'/home/bendel.8/Git_Repos/gaussian_testing/data/stats_{args.d}d/gt_e_vecs.npy')

    # Mac
    # mu_x = np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{args.d}d/gt_mu.npy')
    # e_vals = np.abs(np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{args.d}d/gt_e_vals.npy'))
    # e_vecs = np.load(f'/Users/mattbendel/Documents/pca_gaussian/data/stats_{args.d}d/gt_e_vecs.npy')

    cov_x = e_vecs @ np.diag(e_vals) @ e_vecs.T
    sig_noise = 0.001
    dt = DataTransform(args.d)

    x = torch.from_numpy(np.random.multivariate_normal(mu_x, cov_x, 100))
    x, y, mask = dt(x)

    posterior = Posterior(args.d, mu_x, cov_x, sig_noise, y[-1].numpy())

    best_epoch = 0
    best_cfid = 100000000

    for epoch in range(125, 175):
        print(epoch)
        model = rcGAN.load_from_checkpoint(cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={epoch}.ckpt')
        model.eval().to('cpu')
        posterior_cov_hat = numpy.zeros((args.d, args.d))

        try:
            for i in range(y.shape[0]):
                posterior_mean_hat, posterior_cov_hat_temp = compute_posterior_stats(model, y[i, :].unsqueeze(0), mask, args.d)

                posterior_cov_hat += posterior_cov_hat_temp

            posterior_cov_hat = 1 / y.shape[0] * posterior_cov_hat
            cfid = posterior.cfid(posterior_cov_hat)
            print(cfid)

            if cfid < best_cfid:
                best_epoch = epoch
                best_cfid = cfid
        except Exception as e:
            print(e)
            print('yikes')
            exit()
            continue

    print(best_epoch)
    print(best_cfid)



