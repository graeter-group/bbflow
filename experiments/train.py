# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import GPUtil
import torch

# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
import logging

from gafl import experiment_utils as eu

from bbflow.models.bbflow_module import BBFlowModule
from bbflow.data.pdb_dataloader_bbflow import PdbDataModuleBBFlow

from packaging import version


log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')


class Experiment:

    def __init__(self, *, cfg: DictConfig):
        os.makedirs(cfg.experiment.checkpointer.dirpath, exist_ok=True)
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment

        self.create_data_module()
        self.create_module()

    def create_module(self):
        self._model: LightningModule = BBFlowModule(self._cfg)

    def create_data_module(self):
        self._datamodule: LightningDataModule = PdbDataModuleBBFlow(self._data_cfg)
        
    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._exp_cfg.num_devices = 1
            self._data_cfg.loader.num_workers = 0
        else:
            if self._exp_cfg.use_wandb:
                logger = WandbLogger(
                    **self._exp_cfg.wandb,
                )
            else:
                logger = None
            
            # Checkpoint directory.
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
            callbacks.append(eu.ValidationCallback(self._exp_cfg.first_val_epoch))

            if self._exp_cfg.scheduler.enabled and logger is not None:
                lr_monitor = LearningRateMonitor(logging_interval='epoch')
                callbacks.append(lr_monitor)

            # Save config
            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(eu.flatten_dict(cfg_dict))
            if self._exp_cfg.use_wandb and isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                logger.experiment.config.update(flat_cfg)

        devices = GPUtil.getAvailable(order='first', limit = 8)[:self._exp_cfg.num_devices]
        device_names = [torch.cuda.get_device_name(i) for i in devices]
        log.info(f"Using devices: {devices} ({device_names})")

        trainer_kwargs = dict(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=self._exp_cfg.use_tqdm,
            enable_model_summary=True,
            devices=devices,
        )

        # the `replace_sampler_ddp` flag is deprecated in newer versions of PL
        PL_VERSION = version.parse(pl.__version__)
        if PL_VERSION >= version.parse("2.3.0"):
            trainer_kwargs["use_distributed_sampler"] = False
        else:
            trainer_kwargs["replace_sampler_ddp"] = False

        trainer = Trainer(
            **trainer_kwargs
        )
        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()