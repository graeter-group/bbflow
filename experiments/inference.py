# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""DDP inference script."""
import os
import time
import numpy as np
import hydra
import torch
from torch.utils.data import DataLoader
import GPUtil
from typing import Optional
from pathlib import Path
import pandas as pd
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
import logging

from gafl import experiment_utils as eu

from bbflow.models.bbflow_module import BBFlowModule
from bbflow.data.pdb_dataloader_bbflow import PDBDatasetBBFlowFromPdb

log = eu.get_pylogger(__name__)
logging.getLogger('MDAnalysis.analysis.base').setLevel(logging.WARNING)
logging.getLogger('MDAnalysis.analysis.align').setLevel(logging.WARNING)
torch.set_float32_matmul_precision('high')

class Sampler:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        assert os.path.isfile(ckpt_path), f'Checkpoint path {ckpt_path} does not exist!'
        ckpt_dir = os.path.dirname(ckpt_path)
        cfg_path = os.path.join(ckpt_dir, 'config.yaml')
        assert os.path.isfile(cfg_path), f'Config file {cfg_path} does not exist!'
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'
        cfg.inference.interpolant.prior = ckpt_cfg.interpolant.prior
        # both only necessary if prior != pure_noise
        cfg.inference.interpolant.prior_conditional = ckpt_cfg.interpolant.prior_conditional
        cfg.inference.interpolant.batch_ot = ckpt_cfg.interpolant.batch_ot
        cfg.model.embed = ckpt_cfg.model.embed

        self._cfg = cfg
        self._ckpt_cfg = ckpt_cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        if self._infer_cfg.use_ckpt_name_in_outpath:
            self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
            self._output_dir = os.path.join(
                self._infer_cfg.output_dir,
                self._ckpt_name,
                self._infer_cfg.name,
            )
        else:
            self._output_dir = os.path.join(
                self._infer_cfg.output_dir,
                self._infer_cfg.name,
            )
        os.makedirs(self._output_dir, exist_ok=True)
        config_path = os.path.join(self._output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saving inference config to {config_path}')

        self._flow_module = self.load_module(ckpt_path)
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir

    def load_module(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cuda')
        model_ckpt = ckpt["state_dict"]
        model_ckpt = {k.replace('model.', ''): v for k, v in model_ckpt.items()}

        module = BBFlowModule(self._cfg)
        module.model.load_state_dict(model_ckpt)
        return module

    def run_sampling(self):
        devices = GPUtil.getAvailable(
            order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        device_names = [torch.cuda.get_device_name(i) for i in devices]
        log.info(f"Using devices: {devices} ({device_names})")
        # eval_dataset = eu.LengthDataset(self._samples_cfg)
        
        if self._infer_cfg.csv_path is not None:
            csv = pd.read_csv(self._infer_cfg.csv_path)
            # randomly subsample proteins if num_proteins is set
            if self._infer_cfg.num_proteins is not None:
                csv = csv.sample(
                    n=self._infer_cfg.num_proteins,
                    random_state=self._infer_cfg.seed,
                ).reset_index(drop=True)
            dataset = PDBDatasetBBFlowFromPdb(
                csv, 
                sort=self._infer_cfg.sort,
                min_length=0 if self._infer_cfg.min_length is None else self._infer_cfg.min_length,
                max_length=np.inf if self._infer_cfg.max_length is None else self._infer_cfg.max_length,
            )
        else:
            raise ValueError('No csv path provided')
        
        log.info(f"Sampling {self._infer_cfg.samples.samples_per_protein} conformations per Protein for {len(dataset)} proteins.")
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
        )

        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
            enable_progress_bar=self._infer_cfg.use_tqdm,
        )

        hashtagline = '#' * 80
        log.info(f'\n\n{hashtagline}\n'
                 f'Starting inference run: {self._infer_cfg.name}\nSaving results to {self._output_dir}\n{hashtagline}\n')
        
        trainer.predict(self._flow_module, dataloaders=dataloader)


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = Sampler(cfg)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()