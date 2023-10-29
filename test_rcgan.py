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

    def cfid(self, posterior_mean_hat, posterior_cov_hat):
        # mu_dist = np.linalg.norm(self.posterior_mean - posterior_mean_hat) ** 2
        cov_sum = posterior_cov_hat + self.posterior_cov
        posterior_prod_sqrt = sp.linalg.sqrtm(self.posterior_cov_sqrt @ posterior_cov_hat @ self.posterior_cov_sqrt)

        return np.trace(cov_sum - 2 * posterior_prod_sqrt)


def load_object(dct):
    return types.SimpleNamespace(**dct)


def compute_posterior_stats(generator, y, mask):
    num_z = 1000

    gens = np.zeros((num_z, 10))
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

    model = rcGAN.load_from_checkpoint(cfg.checkpoint_dir + args.exp_name + '/best-mse.ckpt')
    model.eval()

    # model_pca_reg = rcGANwPCAReg.load_from_checkpoint(cfg.checkpoint_dir + args.exp_name + '_pca_reg/best-mse.ckpt')
    # model_pca_reg.eval()
    #
    # model_lazy_reg = rcGANwLazyReg.load_from_checkpoint(cfg.checkpoint_dir + args.exp_name + '_lazy_reg_50_iters/best-mse.ckpt')
    # model_lazy_reg.eval()

    # model_no_std = rcGANNoStdDev.load_from_checkpoint(
    #     cfg.checkpoint_dir + args.exp_name + '_no_std_dev/best-mse.ckpt')
    # model_no_std.eval()

    model_lazy_reg = rcGANwLazyRegSimple.load_from_checkpoint(
        cfg.checkpoint_dir + args.exp_name + '_lazy_reg_P=5_freq=1/best-mse.ckpt')
    model_lazy_reg.eval()

    pca_model = PCANET.load_from_checkpoint(cfg.checkpoint_dir + 'pcanet_gaussian_2/best-pca.ckpt')
    pca_model.eval()

    mu_x = np.load('/Users/mattbendel/Documents/pca_gaussian/data/stats/gt_mu.npy')
    e_vals = np.abs(np.load('/Users/mattbendel/Documents/pca_gaussian/data/stats/gt_e_vals.npy'))
    e_vecs = np.load('/Users/mattbendel/Documents/pca_gaussian/data/stats/gt_e_vecs.npy')

    cov_x = e_vecs @ np.diag(e_vals) @ e_vecs.T
    sig_noise = 0.001

    dt = DataTransform()

    x = torch.from_numpy(np.random.multivariate_normal(mu_x, cov_x, 100))
    x, y, mask = dt(x)

    posterior = Posterior(10, mu_x, cov_x, sig_noise, y[-1].numpy())
    e_vals, e_vecs = np.linalg.eigh(posterior.posterior_cov)

    posterior_cov_hat = numpy.zeros((10, 10))
    posterior_cov_hat_lazy_reg = numpy.zeros((10, 10))
    posterior_cov_hat_no_std = numpy.zeros((10, 10))

    for i in range(y.shape[0]):
        posterior_mean_hat, posterior_cov_hat_temp = compute_posterior_stats(model, y[0, :].unsqueeze(0), mask)
        posterior_mean_hat_lazy_reg, posterior_cov_hat_lazy_reg_temp = compute_posterior_stats(model_lazy_reg, y[0, :].unsqueeze(0), mask)
        # posterior_mean_hat_no_std, posterior_cov_hat_no_std_temp = compute_posterior_stats(model_no_std, y[0, :].unsqueeze(0), mask)

        posterior_cov_hat += posterior_cov_hat_temp
        posterior_cov_hat_lazy_reg += posterior_cov_hat_lazy_reg_temp
        # posterior_cov_hat_no_std += posterior_cov_hat_no_std_temp

    posterior_cov_hat = 1 / y.shape[0] * posterior_cov_hat
    posterior_cov_hat_lazy_reg = 1 / y.shape[0] * posterior_cov_hat_lazy_reg
    posterior_cov_hat_no_std = 1 / y.shape[0] * posterior_cov_hat_no_std

    y = y[0, :].unsqueeze(0)
    x = x[0, :].unsqueeze(0)

    with torch.no_grad():
        x_hat = pca_model.mean_net(y.unsqueeze(0)).unsqueeze(1)
        directions = pca_model.forward(y.unsqueeze(0), x_hat)
        principle_components, diff_vals = pca_model.gramm_schmidt(directions)
        sigma_k = torch.zeros(directions.shape[0], 10).to(directions.device)

        for k in range(directions.shape[1]):
            sigma_k[:, k] = torch.norm(diff_vals[:, k, :], p=2, dim=1) ** 2

    x_hat = x_hat.numpy()
    sigma_k = sigma_k.numpy()

    # swap w/ SVD of data matrix
    _, e_vals_hat, e_vecs_hat = np.linalg.svd(posterior_cov_hat)
    _, e_vals_hat_lazy_reg, e_vecs_hat_lazy_reg = np.linalg.svd(posterior_cov_hat_lazy_reg)
    # _, e_vals_hat_no_std, e_vecs_hat_no_std = np.linalg.svd(posterior_cov_hat_no_std)

    num_samps = 100

    x_axis = np.arange(10) + 1

    # labels = ['True', 'rcGAN + lazy reg']
    labels = ['True', 'rcGAN', 'rcGAN + lazy reg', 'PCANET']
    # labels = ['True', 'rcGAN', 'rcGAN + lazy reg', 'rcGAN + lazy reg - std.', 'PCANET']

    plt.figure()
    plt.scatter(x_axis, posterior.posterior_mean)
    plt.plot(x_axis, posterior_mean_hat)
    plt.plot(x_axis, posterior_mean_hat_lazy_reg)
    # plt.plot(x_axis, posterior_mean_hat_no_std)
    plt.plot(x_axis, x_hat[0, 0, :])
    plt.legend(labels)
    plt.savefig('figs/mean_compare.png')
    plt.close()

    # TODO: Semilogy
    plt.figure()
    plt.scatter(x_axis, np.flip(np.where(e_vals < 1e-3, 0, e_vals)))
    plt.plot(x_axis, np.where(e_vals_hat < 1e-3, 0, e_vals_hat))
    plt.plot(x_axis, np.where(e_vals_hat_lazy_reg < 1e-3, 0, e_vals_hat_lazy_reg))
    # plt.plot(x_axis, np.where(e_vals_hat_no_std < 1e-3, 0, e_vals_hat_no_std))
    plt.plot(x_axis, np.where(sigma_k[0, :] < 1e-3, 0, sigma_k[0, :]))
    plt.legend(labels)
    plt.savefig('figs/eig_val_compare.png')
    plt.close()

    plt.figure()
    plt.scatter([1], np.trace(posterior.posterior_cov))
    plt.scatter([1], np.trace(posterior_cov_hat))
    plt.scatter([1], np.trace(posterior_cov_hat_lazy_reg))
    # plt.scatter([1], np.trace(posterior_cov_hat_no_std))

    pca_cov = np.zeros((10, 10))
    for i in range(principle_components.shape[1]):
        pc_np = np.expand_dims(principle_components[0, 0, :].numpy(), axis=1)
        pca_cov += sigma_k[0, i] * pc_np @ pc_np.T

    temp1, temp2 = np.linalg.eigh(pca_cov)
    print(temp1)
    plt.scatter([1], np.trace(pca_cov))
    plt.legend(labels)
    plt.savefig('figs/trace_compare_rcgan.png')
    plt.close()

    # print("COV DIST")
    # print(f'rcGAN: {posterior.cov_dist(posterior_cov_hat)}')
    # print(f'rcGAN + lazy reg: {posterior.cov_dist(posterior_cov_hat_lazy_reg)}')
    # print(f'PCANET: {posterior.cov_dist(pca_cov)}')

    print("\nCFID")
    print(f'rcGAN: {posterior.cfid(posterior_mean_hat, posterior_cov_hat)}')
    print(f'rcGAN + lazy reg: {posterior.cfid(posterior_mean_hat_lazy_reg, posterior_cov_hat_lazy_reg)}')
    # print(f'rcGAN + lazy reg - std.: {posterior.cfid(posterior_mean_hat_no_std, posterior_cov_hat_no_std)}')
    print(f'PCANET: {posterior.cfid(x_hat, pca_cov)}')

    for i in range(5):
        plt.figure()
        plt.scatter(x_axis, np.abs(e_vecs[:, -(i+1)]))
        plt.plot(x_axis, np.abs(e_vecs_hat[i, :]))
        plt.plot(x_axis, np.abs(e_vecs_hat_lazy_reg[i, :]))
        # plt.plot(x_axis, np.abs(e_vecs_hat_no_std[i, :]))
        plt.plot(x_axis, np.abs(principle_components[0, i].numpy()))
        plt.legend(labels)
        plt.savefig(f'figs/eig_vec_compare_{i}.png')
        plt.close()
