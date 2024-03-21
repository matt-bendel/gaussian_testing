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

    def gramm_schmidt(self, directions):
        principle_components = torch.zeros(directions.shape).to(directions.device)
        diff_vals = torch.zeros(directions.shape).to(directions.device)
        for k in range(directions.shape[1]):
            d = directions[:, k, :].clone()
            if k == 0:
                principle_components[:, 0, :] = d / torch.norm(d, p=2, dim=1)[:, None]
                diff_vals[:, k, :] = d
            else:
                sum_val = torch.zeros(directions.shape[0], self.d).to(directions.device)
                for l in range(k):
                    detached_pc = principle_components[:, l, :].clone().detach()
                    inner_product = torch.sum(d * detached_pc, dim=1)
                    sum_val = sum_val + inner_product[:, None] * detached_pc

                diff_vals[:, k, :] = d - sum_val
                diff_val_clone = diff_vals[:, k, :].clone()
                principle_components[:, k, :] = diff_val_clone / torch.norm(diff_val_clone, p=2, dim=1)[:, None]

        return principle_components, diff_vals

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

        if self.current_epoch >= 20:
            x_hat = x_hat.clone().detach()

            directions = self.forward(y, x_hat)
            principle_components, diff_vals = self.gramm_schmidt(directions)

            sigma_loss = torch.zeros(directions.shape[0]).to(directions.device)
            w_loss = torch.zeros(directions.shape[0]).to(directions.device)
            for k in range(directions.shape[1]):
                e_i_norm = torch.norm((x - x_hat)[:, 0, :], p=2, dim=1)
                w_t_ei = torch.sum(principle_components[:, k, :] * (x - x_hat)[:, 0, :], dim=1)
                w_t_ei_2 = w_t_ei ** 2 / (e_i_norm.clone().detach() ** 2)

                sigma_loss += (torch.norm(diff_vals[:, k, :], p=2, dim=1) ** 2 - w_t_ei.clone().detach() ** 2) ** 2 / (e_i_norm.clone().detach() ** 4)
                w_loss += w_t_ei_2

            sigma_loss = sigma_loss.sum()
            w_loss = - w_loss.sum()

            self.log('sigma_loss', sigma_loss, prog_bar=True)
            self.log('w_loss', w_loss, prog_bar=True)

            pca_loss = w_loss + sigma_loss

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

        if self.current_epoch >= 20:
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

        if self.current_epoch >= 20:
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
