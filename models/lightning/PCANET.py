import torch

import pytorch_lightning as pl
from torch.nn import functional as F

from models.architectures.pca_gens_linear import MMSELinear, PCALinear
from torchmetrics.functional import peak_signal_noise_ratio

class PCANET(pl.LightningModule):
    def __init__(self, args, exp_name, d):
        super().__init__()
        self.args = args
        self.exp_name = exp_name
        self.d = d
        self.second_moment_loss_lambda = 1e-1
        self.second_moment_loss_grace = 200

        self.pca_net = PCALinear(d)
        self.mean_net = MMSELinear(d)

        self.automatic_optimization = False
        self.val_outputs = []

        self.save_hyperparameters()  # Save passed values

    def readd_measures(self, samples, measures, mask):
        samples = (1 - mask) * samples.unsqueeze(1) + measures + 0.0 * torch.ones_like(measures)

        return samples

    def forward(self, y, mean):
        directions = self.pca_net(y, mean)
        return directions

    def gram_schmidt(self, x):
        x_shape = x.shape

        x_orth = []
        proj_vec_list = []
        for i in range(x.shape[1]):
            w = x[:, i, :]
            for w2 in proj_vec_list:
                w = w - w2 * torch.sum(w * w2, dim=-1, keepdim=True)
            w_hat = w.detach() / w.detach().norm(dim=-1, keepdim=True)

            x_orth.append(w)
            proj_vec_list.append(w_hat)

        x_orth = torch.stack(x_orth, dim=1).view(*x_shape)
        return x_orth

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        x, y, mask = batch
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)
        mask = mask.unsqueeze(1)

        opt_mean, opt_pca = self.optimizers()

        x_hat = self.mean_net(y).unsqueeze(1)
        # x_hat = self.readd_measures(x_hat, y, mask)
        mu_loss = F.mse_loss(x_hat, x)

        opt_mean.zero_grad()
        self.manual_backward(mu_loss)
        opt_mean.step()

        self.log('mu_loss', mu_loss, prog_bar=True)

        if self.current_epoch >= 0:
            x_hat = x_hat.clone().detach()

            # directions = self.forward(y, x_hat)
            # principle_components = self.gram_schmidt(directions)

            print('in')
            w_mat = self.gram_schmidt(self.forward(y, x_hat))

            w_mat_ = w_mat.flatten(2)
            w_norms = w_mat_.norm(dim=2)
            w_hat_mat = w_mat_ / w_norms[:, :, None]
            print('out')

            err = (x - x_hat).flatten(1)

            ## Normalizing by the error's norm
            ## -------------------------------
            err_norm = err.norm(dim=1)
            err = err / err_norm[:, None]
            w_norms = w_norms / err_norm[:, None]

            ## W hat loss
            ## ----------
            err_proj = torch.einsum('bki,bi->bk', w_hat_mat, err)
            reconst_err = 1 - err_proj.pow(2).sum(dim=1)

            ## W norms loss
            ## ------------
            second_moment_mse = (w_norms.pow(2) - err_proj.detach().pow(2)).pow(2)

            second_moment_loss_lambda = -1 + 2 * self.global_step / self.second_moment_loss_grace
            second_moment_loss_lambda = max(min(second_moment_loss_lambda, 1), 1e-6)
            second_moment_loss_lambda *= self.second_moment_loss_lambda
            objective = reconst_err.mean() + second_moment_loss_lambda * second_moment_mse.mean()

            #
            # sigma_loss = torch.zeros(directions.shape[0]).to(directions.device)
            # w_loss = torch.zeros(directions.shape[0]).to(directions.device)
            # for k in range(directions.shape[1]):
            #     e_i_norm = torch.norm((x - x_hat)[:, 0, :], p=2, dim=1)
            #     w_t_ei = torch.sum(principle_components[:, k, :] * (x - x_hat)[:, 0, :], dim=1)
            #     w_t_ei_2 = w_t_ei ** 2 / (e_i_norm.clone().detach() ** 2)
            #
            #     sigma_loss += (torch.norm(diff_vals[:, k, :], p=2, dim=1) ** 2 - w_t_ei.clone().detach() ** 2) ** 2 / (e_i_norm.clone().detach() ** 4)
            #     w_loss += w_t_ei_2
            #
            # sigma_loss = sigma_loss.sum()
            # w_loss = - w_loss.sum()

            self.log('sigma_loss', second_moment_mse.mean(), prog_bar=True)
            self.log('w_loss', reconst_err.mean(), prog_bar=True)

            pca_loss = objective

            opt_pca.zero_grad()
            self.manual_backward(pca_loss)
            opt_pca.step()

            self.log('pca_loss', pca_loss, prog_bar=True)


    def validation_step(self, batch, batch_idx, external_test=False):
        x, y, mask = batch
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)
        mask = mask.unsqueeze(1)

        x_hat = self.mean_net(y).unsqueeze(1)
        # x_hat = self.readd_measures(x_hat, y, mask)

        if self.current_epoch >= 10:
            directions = self.forward(y, x_hat)
            principle_components, diff_vals = self.gramm_schmidt(directions)

            psnr_val = peak_signal_noise_ratio(x_hat, x)

            sigma_loss = torch.zeros(directions.shape[0]).to(directions.device)
            w_loss = torch.zeros(directions.shape[0]).to(directions.device)
            for k in range(directions.shape[1]):
                w_t_ei = torch.sum(principle_components[:, k, :] * (x - x_hat)[:, 0, :], dim=1)
                w_t_ei_2 = w_t_ei ** 2

                sigma_loss += (torch.sum(diff_vals[:, k, :] * diff_vals[:, k, :],
                                         dim=1) - w_t_ei.clone().detach() ** 2) ** 2
                w_loss += w_t_ei_2

            sigma_loss = sigma_loss.sum()
            w_loss = - w_loss.sum()

            self.log('w_loss_val', w_loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log('sigma_loss_val', sigma_loss, on_step=True, on_epoch=False, prog_bar=True)

            self.val_outputs.append({'w_val': w_loss, 'psnr_val': psnr_val})

            return {'w_loss_val': w_loss, 'sigma_loss_val': sigma_loss, 'psnr_val': psnr_val}
        else:
            psnr_val = peak_signal_noise_ratio(x_hat, x)

            self.val_outputs.append({'psnr_val': psnr_val})

            return {'psnr_val': psnr_val}

    def on_validation_epoch_end(self):
        psnr_val = torch.stack([x['psnr_val'] for x in self.val_outputs]).mean().mean()
        self.log('psnr_val', psnr_val)

        if self.current_epoch >= 10:
            w_val = torch.stack([x['w_val'] for x in self.val_outputs]).mean().mean()
            self.log('w_val', w_val)
        else:
            self.log('w_val', 10000000)

        self.val_outputs = []

    def on_train_epoch_end(self):
        sch_mean, sch_pca = self.lr_schedulers()

        sch_mean.step(self.trainer.callback_metrics["psnr_val"])
        # if self.current_epoch >= 20:
        #     sch_pca.step(self.trainer.callback_metrics["w_val"])

    def configure_optimizers(self):
        opt_pca = torch.optim.Adam(self.pca_net.parameters(), lr=self.args.lr,
                                 betas=(self.args.beta_1, self.args.beta_2))
        reduce_lr_on_plateau_pca = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_pca,
            mode='max',
            factor=0.1,
            patience=5,
            min_lr=5e-6,
        )

        opt_mean = torch.optim.Adam(self.mean_net.parameters(), lr=self.args.lr,
                                   betas=(self.args.beta_1, self.args.beta_2))
        reduce_lr_on_plateau_mean = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt_mean,
            mode='max',
            factor=0.1,
            patience=5,
            min_lr=5e-6,
        )
        return [[opt_mean, opt_pca], [reduce_lr_on_plateau_mean, reduce_lr_on_plateau_pca]]
