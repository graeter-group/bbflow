# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
from scipy.spatial.transform import Rotation
import copy
import os
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import pandas as pd
import logging
from torch.utils.data.distributed import dist

from gafl.data import so3_utils
from gafl.data import utils as du
from gafl.data import all_atom
from gafl.data.interpolant import (
    Interpolant,
    _centered_gaussian,
    _uniform_so3,
    _trans_diffuse_mask,
    _rots_diffuse_mask
)



class InterpolantBBFlow(Interpolant):

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._rot_distribution = self._rots_cfg.distribution
        self._igso3 = None
        self._log = logging.getLogger(__name__)

        super().__init__(cfg)

    
    def _intra_batch_ot(self, trans_1, trans_0, t, res_mask):
        B, N = trans_0.shape[:2]
        
        if self._cfg.batch_ot == "intra_permute":
    
            trans_0_ordered = torch.zeros_like(trans_0)
            for b in range(B):
                cost_matrix_x = pairwise_distances(
                    trans_1[b].numpy(force=True),
                    trans_0[b].numpy(force=True)
                )

                cost_matrix = cost_matrix_x #+ cost_matrix_r
                _, col_ind = linear_sum_assignment(cost_matrix)

                aligned_0, _, _ = du.batch_align_structures(
                    trans_0[b].clone()[col_ind].unsqueeze(0), 
                    trans_1[b].unsqueeze(0)
                )
                trans_0_ordered[b] = aligned_0[0]

            return trans_0_ordered
        else:
            trans_0, trans_1, _ = du.batch_align_structures(
                trans_0, trans_1, mask=res_mask
            ) 
            return trans_0

    def _corrupt_trans(self, trans_1, trans_0, t, res_mask):

        if self._cfg.batch_ot == "inter":
            trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        elif self._cfg.batch_ot in ["intra", "intra_permute"]:
            trans_0 = self._intra_batch_ot(trans_1, trans_0, t, res_mask)
        elif self._cfg.batch_ot == "none":
            pass
        else:
            raise ValueError(f'Unknown batch ot type {self._cfg.batch_ot}')
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        
        return trans_t * res_mask[..., None]
    
    def _corrupt_rotmats(self, rotmats_1, rotmats_0, t, res_mask):
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch, t=None):
        noisy_batch = copy.deepcopy(batch)

        trans_equilibrium = batch["trans_equilibrium"]
        rotmats_equilibrium = batch["rotmats_equilibrium"]
        B, N = trans_equilibrium.shape[:2]
        trans_0, rotmats_0 = self.sample_from_prior(
            B, N,
            trans_equilibrium,
            rotmats_equilibrium,
        )
        
        if t is None:
            # [B, 1]
            t = self.sample_t(B)[:, None]
        noisy_batch['t'] = t

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        res_mask = batch['res_mask']


        # Apply corruptions
        trans_t = self._corrupt_trans(trans_1, trans_0, t, res_mask)
        noisy_batch['trans_t'] = trans_t

        rotmats_t = self._corrupt_rotmats(rotmats_1, rotmats_0, t, res_mask)
        noisy_batch['rotmats_t'] = rotmats_t
        return noisy_batch
    
    def _pure_noise_prior(self, num_batch, num_res, rotmats_1):
        trans_nm_0 = _centered_gaussian(num_batch, num_res, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        trans_0 *= self.noise_scale

        # make the noise blob larger the longer the protein is:
        if self.noise_res_scaling_power > 0:
            trans_0 *= (trans_0.shape[1] / 128.)**self.noise_res_scaling_power

        if self._rot_distribution == 'uniform':
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        elif self._rot_distribution == 'igso3':
            noisy_rotmats = self.igso3.sample(
                torch.tensor([1.5]),
                num_batch*num_res
            ).to(self._device)
            noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
            rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        else:
            raise ValueError(f'Unknown SO(3) distribution {self._rot_distribution}')
            
        return trans_0, rotmats_0

    def _conditional_prior(self, num_batch, num_res, trans_equilibrium, rotmats_equilibrium):
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        trans_0, rotmats_0 = self._pure_noise_prior(num_batch, num_res, rotmats_equilibrium)

        trans_0 = self._corrupt_trans(
            trans_equilibrium,
            trans_0,
            self._cfg.prior_conditional.gamma_trans * torch.ones(num_batch, 1, device=self._device, dtype=torch.float32),
            res_mask
        )
        rotmats_0 = self._corrupt_rotmats(
            rotmats_equilibrium,
            rotmats_0,
            self._cfg.prior_conditional.gamma_rots * torch.ones(num_batch, 1, device=self._device, dtype=torch.float32),
            res_mask
        )

        return trans_0, rotmats_0

    def sample_from_prior(
            self,
            num_batch,
            num_res,
            trans_equilibrium,
            rotmats_equilibrium,
        ):

        prior_type = self._cfg.prior

        if prior_type == 'conditional':
            return self._conditional_prior(
                num_batch, num_res, trans_equilibrium, rotmats_equilibrium
            )
        elif prior_type == 'pure_noise':
            return self._pure_noise_prior(num_batch, num_res, rotmats_equilibrium)
        
        else:
            raise ValueError(f'Unknown prior type {prior_type}')
    

    def sample(
            self,
            num_batch,
            num_res,
            model,
            trans_equilibrium,
            rotmats_equilibrium,
            seq,
        ):

        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        trans_0, rotmats_0 = self.sample_from_prior(
            num_batch, num_res, trans_equilibrium, rotmats_equilibrium
        )
        batch = {
            "res_mask": res_mask,
            "trans_equilibrium": trans_equilibrium,
            "rotmats_equilibrium": rotmats_equilibrium,
            "seq": seq,
        }

        # Set-up time
        ts = torch.linspace(
            self._cfg.min_t, 1.0, self._sample_cfg.num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        # prot_traj_with_layers = []
        clean_traj = []
        if self._cfg.use_tqdm:
            iterator = tqdm(ts[1:], leave=False, desc=f'Sampling {num_batch} proteins of length {num_res}, rank {dist.get_rank()}')
        else:
            iterator = ts[1:]
        for t_2 in iterator:

            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            batch['t'] = t
            with torch.no_grad():
                model_out = model(batch)
            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch['trans_sc'] = pred_trans_1

            d_t = t_2 - t_1
            # Take reverse step
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2



        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        t = torch.ones((num_batch, 1), device=self._device) * t_1
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = t
        
        with torch.no_grad():
            model_out = model(batch)
        # Process model output.
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )

        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)

        # remove the c betas. they are not predicted by the network but rule based and have no real meaning
        # Masking CB atoms
        # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
        atom37_traj[..., 3, :] = 0
        clean_atom37_traj[..., 3, :] = 0

        return atom37_traj, clean_atom37_traj, clean_traj
    