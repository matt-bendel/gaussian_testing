import torch
import yaml
import types
import json

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from models.lightning.rcGAN import rcGAN
from models.lightning.rcGAN_w_pca_reg import rcGANwPCAReg
from models.lightning.rcGAN_w_lazy_reg import rcGANwLazyReg
from models.lightning.rcGAN_w_lazy_reg_simple_gen import rcGANwLazyRegSimple
from models.lightning.rcGAN_no_std_dev import rcGANNoStdDev
from data.lightning.GaussianDataModule import GaussianDataModule

def load_object(dct):
    return types.SimpleNamespace(**dct)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")

    # if args.rcgan:
    #     with open('configs/rcgan.yml', 'r') as f:
    #         cfg = yaml.load(f, Loader=yaml.FullLoader)
    #         cfg = json.loads(json.dumps(cfg), object_hook=load_object)
    #
    #     model = rcGAN(cfg, args.exp_name)
    # elif args.pcanet:
    with open('configs/rcgan.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    model = rcGAN(cfg, args.exp_name, args.d)
    # model = rcGANwPCAReg(cfg, args.exp_name)
    # model = rcGANwLazyReg(cfg, args.exp_name)
    # model = rcGANwLazyRegSimple(cfg, args.exp_name, args.d)
    print("model")
    # model = rcGANNoStdDev(cfg, args.exp_name)

    dm = GaussianDataModule(args.d)
    print("data")

    # wandb_logger = WandbLogger(
    #     project="pca_exps",
    #     name=args.exp_name,
    #     log_model="all",
    # )
    checkpoint_callback_epoch = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=cfg.checkpoint_dir + args.exp_name + '/',
        filename='best-mse',
        save_top_k=1
    )

    checkpoint_callback_psnr = ModelCheckpoint(
        monitor='psnr_8',
        mode='max',
        dirpath=cfg.checkpoint_dir + args.exp_name + '/',
        filename='best-mse',
        save_top_k=1
    )

    checkpoint_callback_cfid = ModelCheckpoint(
        monitor='cfid',
        mode='min',
        dirpath=cfg.checkpoint_dir + args.exp_name + '/',
        filename='best-mse',
        save_top_k=1
    )

    trainer = pl.Trainer(accelerator="gpu", devices=1, strategy='ddp',
                         max_epochs=cfg.num_epochs, callbacks=[checkpoint_callback_epoch],
                         num_sanity_val_steps=0, profiler="simple", benchmark=False,
                         log_every_n_steps=10)

    # Mac trainer
    # trainer = pl.Trainer(accelerator="mps",
    #                      max_epochs=cfg.num_epochs, callbacks=[checkpoint_callback_epoch],
    #                      num_sanity_val_steps=0, profiler="simple", benchmark=False,
    #                      log_every_n_steps=10)
    trainer.fit(model, dm)
