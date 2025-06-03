# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import os
import time
import numpy as np
import hydra
import torch
from torch.utils.data import DataLoader
import GPUtil
import subprocess
from typing import Optional
from pathlib import Path
import pandas as pd
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from omegaconf.base import ContainerMetadata
import warnings
import logging
from typing import Union
from pathlib import Path
from tqdm.auto import tqdm

# depending on the torch version (higher than 2.x), we need to add the DictConfig and ContainerMetadata to the safe globals:
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([DictConfig, ContainerMetadata]) # needed for loading the checkpoint with weights_only=True

from gafl import experiment_utils as eu
from gafl.data import utils as du
from bbflow.analysis import utils as au
from bbflow.deployment.utils import estimate_max_batchsize, recursive_update, ckpt_path_from_tag

from bbflow.models.bbflow_module import BBFlowModule
from bbflow.data.utils import frames_from_pdb



log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

class BBFlow:

    def __init__(self, ckpt_path: Union[Path,str], cfg:dict={}, timesteps:int=20, gamma_trans:float=None, gamma_rots:float=None, progress_bar:bool=True, _pbar_kwargs={}):
        """
        Arguments:
        ckpt_path: Path or str. Path to checkpoint file. Must contain a 'config.yaml' file in the same directory.
        cfg: dict. default:{}. Configuration dictionary. Will overwrite the configuration from the checkpoint for the entries given.
        timesteps: int or None. default:20. Number of timesteps to sample. Overwrites the configuration from the checkpoint or in cfg.
        gamma_trans: float or None. default:None. Conditional prior parameter controlling how close the prior is to the equilibrium structure. Overwrites the configuration from the checkpoint or in cfg.
        gamma_rots: float or None. default:None. Conditional prior parameter controlling how close the prior is to the equilibrium structure. Overwrites the configuration from the checkpoint or in cfg.
        progress_bar: bool. default:True. Whether to show a progress bar during sampling.
        """
        ckpt_dir = os.path.dirname(ckpt_path)
        config_path = os.path.join(ckpt_dir, 'config.yaml')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {os.path.join(ckpt_dir, 'config.yaml')} does not exist.")
        
        ckpt_cfg = OmegaConf.load(config_path)

        self._pbar_kwargs = _pbar_kwargs


        # Set-up config.
        if timesteps is not None:
            assert isinstance(timesteps, int), "timesteps must be an integer."
            if not 'inference' in cfg:
                cfg['inference'] = {}
            if not 'interpolant' in cfg['inference']:
                cfg['inference']['interpolant'] = {}
            if not 'sampling' in cfg['inference']['interpolant']:
                cfg['inference']['interpolant']['sampling'] = {}
            cfg['inference']['interpolant']['sampling']['num_timesteps'] = timesteps

        default_inference_config = Path(__file__).parent.parent.parent / 'configs/inference.yaml'
        cfg_ = OmegaConf.load(default_inference_config)
        OmegaConf.set_struct(cfg_, True)

        recursive_update(cfg, cfg_)

        OmegaConf.set_struct(cfg_, False)

        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg_, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'

        cfg.inference.interpolant.batch_ot = ckpt_cfg.interpolant.batch_ot
        
        if gamma_rots is None:
            gamma_rots = ckpt_cfg.interpolant.prior_conditional.gamma_rots
        if gamma_trans is None:
            gamma_trans = ckpt_cfg.interpolant.prior_conditional.gamma_trans
            
        cfg.inference.interpolant.prior_conditional.gamma_rots = gamma_rots
        cfg.inference.interpolant.prior_conditional.gamma_trans = gamma_trans
        cfg.model = ckpt_cfg.model

        self._cfg = cfg
        self._cfg.interpolant = cfg.inference.interpolant
        self._cfg.interpolant.use_tqdm = False
        self._ckpt_cfg = ckpt_cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples

        self._progress_bar = progress_bar

        # Read checkpoint and initialize module.
        self._flow_module = self.load_module(ckpt_path)
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg

    @classmethod
    def from_tag(cls, tag:str='latest', cfg:dict={}, timesteps:int=20, gamma_trans:float=None, gamma_rots:float=None, progress_bar:bool=True, _pbar_kwargs={}, force_download:bool=False):
        """
        Arguments:
        tag: str. Tag of the checkpoint to load. Searches the checkpoint at models/{tag}/*.ckpt. If not present, tries to download it.
        cfg: dict. default:{}. Configuration dictionary. Will overwrite the configuration from the checkpoint for the entries given.
        timesteps: int or None. default:None. Number of timesteps to sample. Overwrites the configuration from the checkpoint or in cfg.
        gamma_trans: float or None. default:None. Conditional prior parameter controlling how close the prior is to the equilibrium structure. Overwrites the configuration from the checkpoint or in cfg.
        gamma_rots: float or None. default:None. Conditional prior parameter controlling how close the prior is to the equilibrium structure. Overwrites the configuration from the checkpoint or in cfg.
        progress_bar: bool. default:True. Whether to show a progress bar during sampling.
        _pbar_kwargs: dict. default:{} Additional arguments to pass to the progress bar.
        force_download: bool. default:False. If True, forces the download of the checkpoint even if it is already present.
        """
        ckpt_path = ckpt_path_from_tag(tag, force_download=force_download)
        return cls(ckpt_path, cfg=cfg, timesteps=timesteps, gamma_trans=gamma_trans, gamma_rots=gamma_rots, progress_bar=progress_bar, _pbar_kwargs=_pbar_kwargs)

    def load_module(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=True)
        model_ckpt = ckpt["state_dict"]
        model_ckpt = {k.replace('model.', ''): v for k, v in model_ckpt.items()}

        module = BBFlowModule(self._cfg)
        module.model.load_state_dict(model_ckpt)
        return module
    
    def to(self, device: str):
        self._flow_module.to(device)
        self.device = device

    def _sample_states(self, trans_eq:torch.Tensor, rotmats_eq:torch.Tensor, seq:torch.Tensor, n_samples:int=10, batch_size:int=None, device:str='cuda', cuda_memory_GB:int=40):
        """
        trans_equilibrium: torch.Tensor of shape (n_residues, 3)
        rotmats_equilibrium: torch.Tensor of shape (n_residues, 3, 3)
        seq: torch.Tensor of shape (n_residues, 21)
        n_samples: int
        batch_size: int or None. If not None, overrides the batch size estimation. Otherwise, the batch size is estimated based on the number of residues in the protein.
        device: str. 'cpu' or 'cuda'.
        cuda_memory_GB: int. Maximum amount of memory to use on the GPU. Used for estimating the batch size if batch_size is None.

        Returns:
        atom37_traj: np.array of shape (n_samples, n_residues, 37, 3)
        """
        num_res = trans_eq.shape[0]
        if batch_size is None:
            batch_size = estimate_max_batchsize(n_res=num_res, memory_GB=cuda_memory_GB)
        B = batch_size

        if device != 'cpu':
            assert torch.cuda.is_available(), "CUDA is not available."

        self._flow_module.to(device)
        self._flow_module.interpolant.set_device(device)
        trans_eq = trans_eq.to(device)
        rotmats_eq = rotmats_eq.to(device)
        seq = seq.to(device)

        interpolant = self._flow_module.interpolant
        batches = []
        for i in range(n_samples // B):
            batches.append(B)
        if n_samples % B != 0:
            batches.append(n_samples % B)

        if self._progress_bar:
            progress_bar = tqdm(total=n_samples, desc='Sampling states', dynamic_ncols=True, **self._pbar_kwargs)

        # sample states:
        sampled_conformations = []

        with torch.no_grad():
            for b in batches:
                atom37_traj, pred_traj, _ = interpolant.sample(
                    b,
                    num_res,
                    self._flow_module.model,
                    trans_eq.repeat(b, 1, 1),
                    rotmats_eq.repeat(b, 1, 1, 1),
                    seq.repeat(b, 1, 1),
                )
                atom37_traj = du.to_numpy(torch.stack(atom37_traj, dim=1))

                # remove the c betas. they are not predicted by the network but rule based and have no real meaning
                # Masking CB atoms
                # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
                atom37_traj[..., 3, :] = 0

                for i in range(b):
                    sampled_conformations.append(atom37_traj[i][-1])

                if self._progress_bar:
                    progress_bar.update(b)
        
        # (num_samples, num_residues, 37, 3)
        sampled_conformations = np.stack(sampled_conformations, axis=0)
        
        return sampled_conformations



    def sample(
            self,
            input_path:Union[Path,str],
            num_samples:int=10,
            output_path:Union[Path,str]=None,
            device:str='cuda',
            cuda_memory_GB:int=40,
            batch_size:int=None,
            output_dir:Optional[Union[Path,str]]=None,
            output_fmt:str='pdb',
            overwrite:bool=True
        ):
        """
        Loads a PDB file describing the equilibrium structure of a protein and samples num_samples conformations. Stores the sampled backbone conformations in a PDB file and returns them as array of shape (num_samples, n_residues, 37, 3).

        Arguments:
        input_path: Path or str. Path to PDB file.
        num_samples: int. Number of conformations to sample.
        output_path: Path or str. Path to output PDB file. If None, the sampled conformations are not stored.
        device: str. 'cpu' or 'cuda'.
        cuda_memory_GB: int. Maximum amount of memory to use on the GPU. Used for estimating the batch size if batch_size is None.
        batch_size: int. Batch size for sampling. If None, the batch size is estimated based on the number of residues in the protein.
        output_dir: Path or str. Path to output directory, in which the pdb file with the sampled conformations will be stored as 'sampled_conformations.pdb' if no output_path is given.
        output_fmt: str. 'pdb' or 'xtc'. Format of the output file if output_dir is given.
        overwrite: bool. If True, overwrites the output file if it already exists.
        """

        # some checks:
        assert input_path is not None, "Input path must be given."
        if not str(input_path).endswith('.pdb'):
            raise ValueError(f"Input path must be a PDB file but got {input_path}.")
        
        if output_path is None and output_dir is None:
            raise ValueError("Either output_path or output_dir must be given.")
        
        if output_path is not None and output_dir is not None:
            raise ValueError("Only one of output_path or output_dir must be given.")

        # infer output path:
        if output_dir is not None:
            output_path = Path(output_dir) / f'sampled_conformations.{output_fmt}'

        output_path = Path(output_path)
        
        # print warning if the suffix of the given output path does not match the output format:
        if output_fmt != Path(output_path).suffix[1:]:
            log.warning(f"Output format {output_fmt} does not match the suffix of the given output path {output_path}.")

        if output_path.exists():
            if overwrite:
                warnings.warn(f"An output path already exists and will be overwritten.")
            else:    
                raise ValueError(f"Output path {output_path} already exists.")

        # Load trans/rotmats from PDB
        trans_eq, rotmats_eq, seq, chain_ids = frames_from_pdb(input_path)

        # Sample states
        sampled_conformations = self._sample_states(
            trans_eq=trans_eq,
            rotmats_eq=rotmats_eq,
            seq=seq,
            n_samples=num_samples,
            batch_size=batch_size,
            device=device,
            cuda_memory_GB=cuda_memory_GB
        )

        if str(output_path).endswith('.xtc'):
            raise NotImplementedError("XTC output format is not implemented yet.")
        else:

            logging.info(f"Writing sampled conformations to {output_path}")

            au.write_prot_to_pdb(
                sampled_conformations,
                str(output_path),
                aatype=seq.argmax(-1).numpy(),
                no_indexing=True,
                chain_index=chain_ids
            )