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

    def cfid(self, pmh, pch):
        # mu_dist = np.linalg.norm(self.mu_x[:, 0] - pmh, ord=2) ** 2
        cov_sum = pch + self.posterior_cov
        posterior_prod_sqrt = sp.linalg.sqrtm(self.posterior_cov_sqrt @ pch @ self.posterior_cov_sqrt)

        return np.trace(cov_sum - 2 * posterior_prod_sqrt)


def load_object(dct):
    return types.SimpleNamespace(**dct)


def nmse(x_true, x_fake, x_m):
    return np.linalg.norm(x_true - x_fake, ord=2) ** 2 / (np.linalg.norm(x_true, ord=2) ** 2)


def compute_posterior_stats(generator, y, mask, d):
    num_z = 1000

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

    model = rcGAN.load_from_checkpoint(cfg.checkpoint_dir + f'rcgan_d={args.d}/best.ckpt')
    model.eval().to('cpu')

    # model_pca_reg = rcGANwPCAReg.load_from_checkpoint(cfg.checkpoint_dir + args.exp_name + '_pca_reg/best-mse.ckpt')
    # model_pca_reg.eval()
    #
    # model_lazy_reg = rcGANwLazyReg.load_from_checkpoint(cfg.checkpoint_dir + args.exp_name + '_lazy_reg_50_iters/best-mse.ckpt')
    # model_lazy_reg.eval()

    # model_no_std = rcGANNoStdDev.load_from_checkpoint(
    #     cfg.checkpoint_dir + args.exp_name + '_no_std_dev/best-mse.ckpt')
    # model_no_std.eval()

    model_lazy_reg = rcGANwLazyRegSimple.load_from_checkpoint(cfg.checkpoint_dir + f'rcgan_lazy_d={args.d}/best.ckpt')
        # rcGANwLazyRegSimple.load_from_checkpoint(
        # cfg.checkpoint_dir + 'rcgan_gaussian_reg_d=60_freq=100/best.ckpt')
    model_lazy_reg.eval().to('cpu')

    pca_model = PCANET.load_from_checkpoint(cfg.checkpoint_dir + f'nppc_d{args.d}/best-pca.ckpt')
    pca_model.eval()

    mu_x = np.load(f'/home/bendel.8/Git_Repos/gaussian_testing/data/stats_{args.d}d/gt_mu.npy')
    e_vals = np.abs(np.load(f'/home/bendel.8/Git_Repos/gaussian_testing/data/stats_{args.d}d/gt_e_vals.npy'))
    e_vecs = np.load(f'/home/bendel.8/Git_Repos/gaussian_testing/data/stats_{args.d}d/gt_e_vecs.npy')

    cov_x = e_vecs @ np.diag(e_vals) @ e_vecs.T
    sig_noise = 0.001
    dt = DataTransform(args.d)

    x = torch.from_numpy(np.random.multivariate_normal(mu_x, cov_x, 100))
    x, y, mask = dt(x)

    posterior = Posterior(args.d, mu_x, cov_x, sig_noise, y[-1].numpy())
    e_vals, e_vecs = np.linalg.eigh(posterior.posterior_cov)

    posterior_cov_hat = numpy.zeros((args.d, args.d))
    posterior_cov_hat_lazy_reg = numpy.zeros((args.d, args.d))

    posterior_means = np.zeros((100, args.d))
    posterior_means_reg = np.zeros((100, args.d))

    nmses = []
    nmses_reg = []

    for i in range(y.shape[0]):
        posterior_mean_hat, posterior_cov_hat_temp = compute_posterior_stats(model, y[i, :].unsqueeze(0), mask, args.d)
        posterior_mean_hat_lazy_reg, posterior_cov_hat_lazy_reg_temp = compute_posterior_stats(model_lazy_reg, y[i, :].unsqueeze(0), mask, args.d)

        nmses.append(nmse(x[i, :].numpy(), posterior_mean_hat, np.mean(x.numpy(), axis=0)))
        nmses_reg.append(nmse(x[i, :].numpy(), posterior_mean_hat_lazy_reg, np.mean(x.numpy(), axis=0)))

        posterior_means[i, :] = posterior_mean_hat
        posterior_means_reg[i, :] = posterior_mean_hat_lazy_reg

        posterior_cov_hat += posterior_cov_hat_temp
        posterior_cov_hat_lazy_reg += posterior_cov_hat_lazy_reg_temp

    posterior_mean_hat, _ = compute_posterior_stats(model, y[-1, :].unsqueeze(0), mask, args.d)
    posterior_mean_hat_lazy_reg, _ = compute_posterior_stats(model_lazy_reg, y[-1, :].unsqueeze(0), mask, args.d)

    posterior_cov_hat = 1 / y.shape[0] * posterior_cov_hat
    posterior_cov_hat_lazy_reg = 1 / y.shape[0] * posterior_cov_hat_lazy_reg

    y = y[0, :].unsqueeze(0)
    x = x[0, :].unsqueeze(0)

    with torch.no_grad():
        x_hat = pca_model.mean_net(y.unsqueeze(0)).unsqueeze(1)
        print(x_hat.shape)
        print(y.unsqueeze(0).shape)
        directions = pca_model.forward(y.unsqueeze(0), x_hat)
        w_mat = pca_model.gram_schmidt(pca_model.forward(y, x_hat))

        w_mat_ = w_mat.flatten(2)
        w_norms = w_mat_.norm(dim=2)
        principle_components = w_mat_ / w_norms[:, :, None]
        sigma_k = w_norms ** 2

    x_hat = x_hat.numpy()
    sigma_k = sigma_k.numpy()

    # swap w/ SVD of data matrix
    _, e_vals_hat, e_vecs_hat = np.linalg.svd(posterior_cov_hat)
    _, e_vals_hat_lazy_reg, e_vecs_hat_lazy_reg = np.linalg.svd(posterior_cov_hat_lazy_reg)
    # _, e_vals_hat_no_std, e_vecs_hat_no_std = np.linalg.svd(posterior_cov_hat_no_std)

    num_samps = 100

    x_axis = np.arange(args.d) + 1

    # labels = ['True', 'rcGAN + lazy reg']
    labels = ['True', 'rcGAN', 'rcGAN + lazy reg']
    # labels = ['True', 'rcGAN', 'rcGAN + lazy reg', 'rcGAN + lazy reg - std.', 'PCANET']

    plt.figure()
    plt.scatter(x_axis, posterior.posterior_mean)
    print(np.mean((posterior.posterior_mean[:, 0] - posterior_mean_hat)**2))
    print(np.mean((posterior.posterior_mean[:, 0] - posterior_mean_hat_lazy_reg) ** 2))

    plt.plot(x_axis, posterior_mean_hat)
    plt.plot(x_axis, posterior_mean_hat_lazy_reg)
    # plt.plot(x_axis, posterior_mean_hat_no_std)
    # plt.plot(x_axis, x_hat[0, 0, :])
    plt.legend(labels)
    plt.savefig(f'figs/mean_compare_{args.d}.png')
    plt.close()

    # TODO: Semilogy
    plt.figure()
    plt.scatter(x_axis, np.flip(np.where(e_vals < 1e-3, 0, e_vals)))
    plt.plot(x_axis, np.where(e_vals_hat < 1e-3, 0, e_vals_hat))
    plt.plot(x_axis, np.where(e_vals_hat_lazy_reg < 1e-3, 0, e_vals_hat_lazy_reg))
    # plt.plot(x_axis, np.where(e_vals_hat_no_std < 1e-3, 0, e_vals_hat_no_std))
    # plt.plot(x_axis, np.where(sigma_k[0, :] < 1e-3, 0, sigma_k[0, :]))
    plt.legend(labels)
    plt.savefig(f'figs/eig_val_compare_{args.d}.png')
    plt.close()

    plt.figure()
    plt.scatter([1], np.trace(posterior.posterior_cov))
    plt.scatter([1], np.trace(posterior_cov_hat))
    plt.scatter([1], np.trace(posterior_cov_hat_lazy_reg))
    # plt.scatter([1], np.trace(posterior_cov_hat_no_std))

    pca_cov = np.zeros((args.d, args.d))
    for i in range(principle_components.shape[1]):
        pc_np = np.expand_dims(principle_components[0, i, :].numpy(), axis=1)
        pca_cov += sigma_k[0, i] * pc_np @ pc_np.T

    # temp1, temp2 = np.linalg.eigh(pca_cov)
    # print(temp1)
    # plt.scatter([1], np.trace(pca_cov))
    plt.legend(labels)
    plt.savefig(f'figs/trace_compare_rcgan_{args.d}.png')
    plt.close()

    # print("COV DIST")
    # print(f'rcGAN: {posterior.cov_dist(posterior_cov_hat)}')
    # print(f'rcGAN + lazy reg: {posterior.cov_dist(posterior_cov_hat_lazy_reg)}')
    # print(f'PCANET: {posterior.cov_dist(pca_cov)}')

    print("\nCFID")
    # rcgan_cfids = posterior.cfid(np.mean(posterior_means, axis=0), posterior_cov_hat)
    # rcgan_w_reg_cfids = posterior.cfid(np.mean(posterior_means_reg, axis=0), posterior_cov_hat_lazy_reg)
    # print(f'rcGAN: m={np.mean((posterior.posterior_mean[:, 0] - posterior_mean_hat)**2)}, c:{rcgan_cfids[1]}')
    # print(f'rcGAN + lazy reg: m={np.mean((posterior.posterior_mean[:, 0] - posterior_mean_hat_lazy_reg) ** 2)}, c:{rcgan_w_reg_cfids[1]}')
    # print(f'rcGAN + lazy reg - std.: {posterior.cfid(posterior_mean_hat_no_std, posterior_cov_hat_no_std)}')
    print(f'PCANET: {posterior.cfid(x_hat, pca_cov)}')

    for i in range(5):
        plt.figure()
        # plt.scatter(x_axis, np.abs(e_vecs[:, -(i+1)]))
        plt.plot(x_axis, np.abs(np.abs(e_vecs_hat[i, :]) - np.abs(e_vecs[:, -(i+1)])))
        plt.plot(x_axis, np.abs(np.abs(e_vecs_hat_lazy_reg[i, :]) - np.abs(e_vecs[:, -(i+1)])))
        # plt.plot(x_axis, np.abs(e_vecs_hat_no_std[i, :]))
        # plt.plot(x_axis, np.abs(principle_components[0, i].numpy()))
        plt.legend(['rcGAN', 'rcGAN + lazy reg'])
        plt.title(f'Eigenvector Error {i+1}')
        plt.savefig(f'figs/eig_vec_compare_{args.d}_{i}.png')
        plt.close()

    print(f'NMSE rcGAN: {np.mean(nmses)}')
    print(f'NMSE rcGAN + reg: {np.mean(nmses_reg)}')
