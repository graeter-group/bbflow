# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import shutil
import json
from datetime import datetime
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data.distributed import dist
import matplotlib
import mdtraj
import warnings

from gafl.analysis import metrics 
from gafl.models.flow_module import FlowModule
from gafl.data import utils as du
import gafl.experiment_utils as eu
from torch.utils.data.distributed import dist

from bbflow.models.bbflow_model import BBFlowModel
from bbflow.analysis import utils as au
from bbflow.analysis.analyse_bbflow import calc_metrics
from bbflow.data.interpolant_bbflow import InterpolantBBFlow


class BBFlowModule(FlowModule):

    def __init__(self, cfg, folding_cfg=None):
        super().__init__(cfg, folding_cfg)
        self._print_logger = eu.get_pylogger(__name__)
        if hasattr(self._exp_cfg, 'reset_epoch'):
            self.reset_epoch = self._exp_cfg.reset_epoch
        else:
            self.reset_epoch = False

        # set samples per valid protein:
        num_ranks = cfg.experiment.num_devices
        total_samples_per_valid_prot = self._data_cfg.validation.samples_per_valid_protein
        self.samples_per_valid_protein_per_rank = total_samples_per_valid_prot // num_ranks

        assert self.samples_per_valid_protein_per_rank % num_ranks == 0, f"Samples per valid protein ({self.samples_per_valid_protein_per_rank}) must be divisible by the number of ranks ({num_ranks})"
        assert self.samples_per_valid_protein_per_rank > 0


    def create_model(self):
        self.model = BBFlowModel(self._model_cfg)
        self.interpolant = InterpolantBBFlow(self._interpolant_cfg)


    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int=0):
        pdb_name = batch["pdb_name"][0]

        epoch_dir = os.path.join(self._sample_write_dir, f'epoch_{self.current_epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        pdb_dir = os.path.join(epoch_dir, pdb_name)
        os.makedirs(pdb_dir, exist_ok=True)
        
        N = batch['seq'].shape[1]
        batch_size = self._data_cfg.validation.max_valid_num_res_squared / N**2
        batch_size = max(
            1,
            min(
                self._data_cfg.validation.max_valid_batch_size,
                int(batch_size/5)*5
            )
        )
        self.interpolant.set_device(batch["trans_equilibrium"].device)
        # print(self.interpolant._cfg)
        sampled_conformations = self._sample_multiple(
            batch["trans_equilibrium"],
            batch["rotmats_equilibrium"],
            batch["seq"],
            self.samples_per_valid_protein_per_rank,
            batch_size,
            self.interpolant,
        )


        rank = dist.get_rank()
        if dist.get_world_size() > 1:
            # if multi-gpu, only save npy files and combine in on_validation_epoch_end
            # into one pdb file.
            samples_path = os.path.join(pdb_dir, f'rank_{rank}_sampled_conformations.npy')
            np.save(samples_path, sampled_conformations)
        else:
            # if single-gpu, save pdb file directly
            samples_path = os.path.join(pdb_dir, f'sampled_conformations.pdb')
            au.write_prot_to_pdb(
                sampled_conformations,
                samples_path,
                no_indexing=True
            )


    def on_validation_epoch_end(self):
        # wait until all ranks finished validation_step
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()  # wait until all ranks wrote their .npy

        if dist.is_initialized() and dist.get_rank() != 0:
            return  # only rank 0 continues

        # make image saving faster:
        matplotlib.use("Agg")   # non-interactive, fast, no GUI needed
        import matplotlib.pyplot as plt 

        folder = os.path.join(
            self._sample_write_dir,
            f'epoch_{self.current_epoch}',
        )
        valid_csv_path = os.path.join(self._sample_write_dir, 'valid.csv')
        valid_csv = pd.read_csv(valid_csv_path)
        all_pdb_names = valid_csv['pdb_name'].values
        pdb_names = []
        if dist.get_world_size() > 1:
            # combine samples that are saved in npy files into one pdb file
            for pdb_name in all_pdb_names:
                pdb_folder = os.path.join(folder, pdb_name)
                if self.trainer.sanity_checking and not os.path.exists(pdb_folder):
                    continue
                pdb_names.append(pdb_name)

                coords = []
                for rank in range(dist.get_world_size()):
                    samples_path = os.path.join(pdb_folder, f'rank_{rank}_sampled_conformations.npy')
                    coords.append(np.load(samples_path))
                    os.remove(samples_path)
                coords = np.concatenate(coords, axis=0)
                
                samples_path = os.path.join(pdb_folder, f'sampled_conformations.pdb')
                au.write_prot_to_pdb(
                    coords,
                    samples_path,
                    no_indexing=True
                )
        else:
            for pdb_name in all_pdb_names:
                pdb_folder = os.path.join(folder, pdb_name)
                if self.trainer.sanity_checking and not os.path.exists(pdb_folder):
                    continue
                pdb_names.append(pdb_name)



        if self.trainer.sanity_checking or not self._exp_cfg.use_wandb:
            return

        metrics_subset = self._data_cfg.validation.metrics_subset
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            data, metrics_dict, _ = calc_metrics(
                pdb_names,
                self._data_cfg.validation.valid_xtc_paths,
                folder,
                137,
                self._data_cfg.loader.num_workers,
                metrics_subset=metrics_subset,
                use_tqdm=self._exp_cfg.use_tqdm
            )
        for key, value in metrics_dict.items():
            self._log_scalar(
                f'valid_bbf_metrics/{key}', value,
                on_step=False, on_epoch=True, prog_bar=False,
            )


        af_rmsfs = None
        if self._data_cfg.validation.alphaflow_valid_rmsf_path is not None and os.path.exists(self._data_cfg.validation.alphaflow_valid_rmsf_path):
            with open(self._data_cfg.validation.alphaflow_valid_rmsf_path, 'r') as f:
                af_rmsfs = json.load(f)
        else:
            af_rmsfs = None

        def _plot_rmsf(pdb_name, ax, md, bbf, af=None, rmsf_type="RMSF"):
            rmse = np.sqrt(np.mean((bbf[0] - md[0])**2))
            xs = np.arange(len(md[0]))
            ax.plot(xs, md[0], label="MD", c="C0")
            ax.fill_between(xs, md[1], md[2], color="C0", alpha=0.5)
            ax.plot(xs, bbf[0], label="BBFlow", c="C1")
            ax.fill_between(xs, bbf[1], bbf[2], color="C1", alpha=0.5)
            if af is not None:
                ax.plot(xs, af["rmsf"], label="AlphaFlow", c="C2")
                ax.fill_between(xs, af["low"], af["high"], color="C2", alpha=0.5)
            ax.set_xlabel("Residue")
            ax.set_ylabel(f"{rmsf_type} "+r"[$\AA$]")
            ax.legend()
            ax.set_title(f"{rmsf_type} {pdb_name}, Epoch {self.current_epoch}, RMSE={rmse:.3f}")

        if self._exp_cfg.use_wandb and "RMSF" in metrics_subset:
            SMALL_SIZE = 20
            MEDIUM_SIZE = 24
            BIGGER_SIZE = 28
            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

            for pdb_name in pdb_names:
                fig = plt.figure(figsize=(10, 5))
                _plot_rmsf(
                    pdb_name, plt.gca(),
                    (data[pdb_name]["CA_ref_rmsf"], data[pdb_name]["CA_ref_rmsf_low"], data[pdb_name]["CA_ref_rmsf_high"]),
                    (data[pdb_name]["CA_model_rmsf"], data[pdb_name]["CA_model_rmsf_low"], data[pdb_name]["CA_model_rmsf_high"]),
                    af_rmsfs[pdb_name] if af_rmsfs is not None else None,
                    "RMSF"
                )
                wandb.log({f"rmsf_plots/{pdb_name}": wandb.Image(fig)})
                plt.close(fig)


        if "RMSF" in metrics_subset:
            rmses = []
            for pdb_name in pdb_names:
                rmse = np.sqrt(np.mean((data[pdb_name]["model_rmsf"] - data[pdb_name]["ref_rmsf"])**2))
                rmses.append(rmse)
                self._log_scalar(
                    f'valid_all/{pdb_name}_rmsf_rmse', rmse,
                    on_step=False, on_epoch=True, prog_bar=False,
                )

            rmses = np.array(rmses)
            self._log_scalar(
                f'valid/rmsf_rmse', np.mean(rmses),
                on_step=False, on_epoch=True, prog_bar=False,
            )

    

    def _sample_multiple(self, 
        trans_eq, rotmats_eq, seq, 
        num_samples, batch_size,
        interpolant,
        pdb_dir=None
    ):
        _, N = trans_eq.shape[:2]
        diffuse_mask = torch.ones(1, N)

        sample_ids_batches = np.split(
            np.arange(num_samples),
            np.arange(batch_size, num_samples, batch_size)
        )

        sampled_conformations = []
        
        for sample_ids in sample_ids_batches:
            B = len(sample_ids)
            atom37_traj, pred_traj, _ = interpolant.sample(
                B,
                N,
                self.model,
                trans_eq.repeat(B, 1, 1),
                rotmats_eq.repeat(B, 1, 1, 1),
                seq.repeat(B, 1, 1),
            )

            atom37_traj = du.to_numpy(torch.stack(atom37_traj, dim=1))
            pred_traj = du.to_numpy(torch.stack(pred_traj, dim=1))
            torch.cuda.empty_cache()
            # B, T, N, 37, 3

            # remove the c betas. they are not predicted by the network but rule based and have no real meaning
            # Masking CB atoms
            # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
            atom37_traj[..., 3, :] = 0
            pred_traj[..., 3, :] = 0

            for i, sample_id in enumerate(sample_ids):
                sampled_conformations.append(atom37_traj[i, -1])

                if pdb_dir is not None:
                    sample_dir = os.path.join(pdb_dir, f'conf_{sample_id}')
                    os.makedirs(sample_dir, exist_ok=True)
                    eu.save_traj(
                        atom37_traj[i, -1],
                        np.flip(atom37_traj[i], axis=0),
                        np.flip(pred_traj[i], axis=0),
                        du.to_numpy(diffuse_mask)[0],
                        output_dir=sample_dir,
                    )

        sampled_conformations = np.stack(sampled_conformations, axis=0)
        return sampled_conformations
    
    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        # If using a Gafl checkpoint, don't use weights of node_embedder and edge_embedder
        new_state_dict = {}
        for k,v in checkpoint["state_dict"].items():
            if 'node_embedder' in k:
                continue
            if 'edge_embedder' in k:
                continue
            new_state_dict[k] = v
        for k,v in self.model.state_dict().items():
            if f"model.{k}" not in new_state_dict:
                new_state_dict[f"model.{k}"] = v
        checkpoint['state_dict'] = new_state_dict
            
        if self.reset_optimizer_on_load:
            optimizers = self.configure_optimizers()
            if isinstance(optimizers, tuple):
                optimizers = optimizers[0]
                lr_schedulers = optimizers[1]
                checkpoint['optimizer_states'] = [o.state_dict() for o in optimizers]
                checkpoint['lr_schedulers'] = [s.state_dict() for s in lr_schedulers]
            else:
                checkpoint['optimizer_states'] = [optimizers.state_dict()]

        if self.reset_epoch:
            checkpoint["epoch"] = 0
            checkpoint["global_step"] = 0
            del checkpoint["loops"]
    

    def predict_step(self, batch, batch_idx):
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = InterpolantBBFlow(self._infer_cfg.interpolant) 
        interpolant.set_device(device)
        
        pdb_dir = os.path.join(self._output_dir, batch["pdb_name"][0])
        if os.path.exists(os.path.join(pdb_dir, f'rmsf.png')):
            print(f"Skipping {batch['pdb_name']} as it already exists")
            return
        os.makedirs(pdb_dir, exist_ok=True)

        sample_time_start = datetime.now()

        N = batch["trans_equilibrium"].shape[1]
        batch_size = self._infer_cfg.samples.num_res_squared / N**2
        batch_size = max(
            1,
            min(
                self._infer_cfg.samples.batch_size,
                int(batch_size/5)*5
            )
        )

        sampled_conformations = self._sample_multiple(
            batch["trans_equilibrium"],
            batch["rotmats_equilibrium"],
            batch["seq"],
            self._infer_cfg.samples.samples_per_protein,
            batch_size,
            interpolant,
            pdb_dir=pdb_dir if self._infer_cfg.save_traj else None
        )

        sample_time = (datetime.now()-sample_time_start).total_seconds()

        aatype = batch["seq"][0].argmax(-1).cpu().numpy()

        samples_path = os.path.join(pdb_dir, f'sampled_conformations.pdb')
        au.write_prot_to_pdb(
            sampled_conformations,
            samples_path,
            aatype=aatype,
            no_indexing=True
        )

        confs = mdtraj.load(samples_path)
        avg = np.mean(confs.xyz, axis=0)
        avg -= np.mean(avg[confs.top.select("name CA")], axis=0, keepdims=True)
        avg = mdtraj.Trajectory(avg[None, :, :], confs.top)
        confs = confs.superpose(avg)
        rmsf = mdtraj.rmsf(confs, avg)
        confs.save_pdb(samples_path, bfactors=rmsf)


        if not self._infer_cfg.single_file:
            os.makedirs(os.path.join(pdb_dir, "samples"))
            confs = mdtraj.load(samples_path)
            for i in range(len(confs)):
                single_sample_path = os.path.join(pdb_dir, "samples", f"sample_{i}.pdb")
                confs[i].save(single_sample_path)

            os.remove(samples_path)


        if self._infer_cfg.log_times:
            self._print_logger.info(f"Generating time for {batch['pdb_name']} for {self._infer_cfg.samples.samples_per_protein} samples of length {N}: {sample_time}s")
            sample_time = (datetime.now()-sample_time_start).total_seconds()
            self._print_logger.info(f"Time with pdb saving and plotting: {sample_time}s")