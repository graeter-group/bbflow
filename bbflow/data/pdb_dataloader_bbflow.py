# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) 2024 HITS gGmbH
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""PDB data loader."""
import math
import torch
import tree
import numpy as np
import torch
import pandas as pd
import logging
import json
import os
from tqdm import tqdm
import shutil
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist
import mdtraj

from gafl.data.residue_constants import restype_order, restype_3to1, restypes

# from bbflow.analysis.analyse_bbflow import calc_aligned_rmsf
from bbflow.analysis.analyse_bbflow import _calc_rmsf, _align_tops

from bbflow.data.utils import frames_from_pdb, backbone_to_frames



def get_alphaflow_valid_rmsf(pdb_path, rmsf_file_name, xtc_path, valid_csv):
    # Precompute RMSF for validation proteins for AlphaFlow
    # Used for plotting rmsf-profiles in validation
    af_valid_pdb_folder = os.path.abspath(pdb_path)
    af_valid_rmsf_precomputed_path = os.path.abspath(os.path.join(
        pdb_path, rmsf_file_name
    ))
    if os.path.exists(af_valid_rmsf_precomputed_path):
        with open(af_valid_rmsf_precomputed_path, "r") as f:
            rmsfs = json.load(f)
    else:
        rmsfs = {}
        for i, row in tqdm(valid_csv.iterrows(), total=len(valid_csv), desc="calculate rmsf for alphaflow valid"):
            pdb_name = row['pdb_name']
            af_pdb_name = f"{pdb_name[:4]}_{pdb_name[4]}"
            if not os.path.exists(os.path.join(af_valid_pdb_folder, f"{af_pdb_name}.pdb")):
                af_pdb_name = pdb_name
            ref = mdtraj.load(os.path.join(xtc_path, f"{pdb_name}/{pdb_name}.pdb"))
            af_traj = mdtraj.load(os.path.join(af_valid_pdb_folder, f"{af_pdb_name}.pdb"))
            ref.atom_slice([a.index for a in ref.top.atoms if a.element.symbol != 'H'], True)
            af_traj.atom_slice([a.index for a in af_traj.top.atoms if a.element.symbol != 'H'], True)
            af_traj.atom_slice([a.index for a in af_traj.top.atoms if a.name in ['CA', 'C', 'N', 'O', 'OXT']], True)
            refmask, afmask = _align_tops(ref.top, af_traj.top, True)
            ref.atom_slice(refmask, True)
            af_traj.atom_slice(afmask, True)
            ca_mask = [a.index for a in ref.top.atoms if a.name == 'CA']
            ca_mask_model = [a.index for a in af_traj.top.atoms if a.name == 'CA']
            ref = ref.atom_slice(ca_mask, False)
            af_traj = af_traj.atom_slice(ca_mask_model, False)
            af_traj.superpose(ref)
            rmsf = _calc_rmsf(af_traj, ref, bootstrap=True)
            rmsfs[pdb_name] = {
                "rmsf": rmsf["rmsf"].tolist(),
                "low": rmsf["rmsf_low"].tolist(),
                "high": rmsf["rmsf_high"].tolist(),
            }
        with open(af_valid_rmsf_precomputed_path, "w") as f:
            json.dump(rmsfs, f)

    return rmsfs

    

class PdbDataModuleBBFlow(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.data_cfg = data_cfg
        
        self.loader_cfg = data_cfg.loader
        self.dataset_cfg = data_cfg.dataset
        self.conditional_val_cfg = data_cfg.validation
        self.sampler_cfg = data_cfg.sampler

        self.train_csv = pd.read_csv(self.dataset_cfg.train_csv_path)
        self.train_csv = self.train_csv[self.train_csv.modeled_seq_len <= self.dataset_cfg.max_num_res]
        self.train_csv = self.train_csv[self.train_csv.modeled_seq_len >= self.dataset_cfg.min_num_res]
        self.train_csv = self.train_csv.sort_values('modeled_seq_len', ascending=True).reset_index(drop=True)
        
        self.valid_csvs = []
        for i, valid_csv_path in enumerate(self.dataset_cfg.valid_csv_paths):
            valid_csv = pd.read_csv(valid_csv_path)
            # valid_csv = valid_csv[valid_csv.modeled_seq_len <= self.dataset_cfg.max_num_res]
            # valid_csv = valid_csv[valid_csv.modeled_seq_len >= self.dataset_cfg.min_num_res]
            # valid_csv = valid_csv.sort_values('modeled_seq_len', ascending=False).reset_index(drop=True)
            valid_csv.insert(0, 'dataset_name', self.dataset_cfg.valid_dataset_names[i])
            self.valid_csvs.append(valid_csv)
        

        self.test_csv = pd.read_csv(self.dataset_cfg.test_csv_path)
        self.test_csv = self.test_csv[self.test_csv.modeled_seq_len <= self.dataset_cfg.max_num_res]
        self.test_csv = self.test_csv[self.test_csv.modeled_seq_len >= self.dataset_cfg.min_num_res]
        self.test_csv = self.test_csv.sort_values('modeled_seq_len', ascending=True).reset_index(drop=True)

        if self.dataset_cfg.subset is not None:
            self.train_csv = self.train_csv.iloc[:self.dataset_cfg.subset]
            for i, valid_csv in enumerate(self.valid_csvs):
                self.valid_csvs[i] = valid_csv.iloc[:self.dataset_cfg.subset]

        self.train_csv.to_csv(os.path.join(self.dataset_cfg.split_path, 'train.csv'), index=False)
        self.test_csv.to_csv(os.path.join(self.dataset_cfg.split_path, 'test.csv'), index=False)
        pd.concat(self.valid_csvs).to_csv(os.path.join(self.dataset_cfg.split_path, 'valid.csv'), index=False)


        if self.conditional_val_cfg.alphaflow_valid_rmsf_path is not None:
            # if the path is not None, rmsf for alphaflow's validation set ensembles are precomputed
            # This is used in the validation step to plot the rmsf-profiles together with MD rmsf and
            # bbflow's rmsf using wandb.
            af_rmsf = {}
            for i in range(len(self.valid_csvs)):
                valid_csv = self.valid_csvs[i]
                xtc_path = self.conditional_val_cfg.valid_xtc_paths[i]
                af_pdb_path = self.conditional_val_cfg.alphaflow_valid_pdb_paths[i]
                af_rmsf_i = get_alphaflow_valid_rmsf(
                    af_pdb_path,
                    os.path.basename(self.conditional_val_cfg.alphaflow_valid_rmsf_path),
                    xtc_path,
                    valid_csv
                )
                af_rmsf.update(af_rmsf_i)
                
            af_valid_rmsf_path = os.path.abspath(self.conditional_val_cfg.alphaflow_valid_rmsf_path)
            with open(af_valid_rmsf_path, "w") as f:
                json.dump(af_rmsf, f)


            

    def setup(self, stage: str):
        self._log.info(f'Training: {len(self.train_csv)} examples')
        self._train_dataset = PDBDatasetBBFlow(
            csv=self.train_csv,
        )
        self._log.info(f"Validation: {sum([len(valid_csv) for valid_csv in self.valid_csvs])} examples")
        self._valid_datasets = [
            PDBDatasetBBFlowFromPdb(valid_csv, sort=True)
            for valid_csv in self.valid_csvs
        ]

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        batch_sampler = LengthBatcherBBFlow(
            sampler_cfg=self.sampler_cfg,
            metadata_csv=self._train_dataset.csv,
            rank=rank,
            num_replicas=num_replicas,
        )
        return DataLoader(
            self._train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )

    def val_dataloader(self):
        valid_dataloader = [
            DataLoader(
                valid_dataset,
                shuffle=False,
                num_workers=2,
                prefetch_factor=2,
                persistent_workers=True,
            )
            for valid_dataset in self._valid_datasets
        ]

        return valid_dataloader


class PDBDatasetBBFlow(Dataset):
    def __init__(self, csv):
        self._log = logging.getLogger(__name__)
        self.csv = csv

        self._log.info(f'Training: {len(self.csv)} examples')

        self._init_data()

    def _init_data(self):
        """
        Preprocess equilibrium structures for training.
        """
        num_proteins = len(self.csv)
        max_length = self.csv["modeled_seq_len"].max()
        self.num_proteins = num_proteins
        self.max_protein_length = max_length

        self.trans_equilibrium = []
        self.rotmats_equilibrium = []
        self.seq_1letter = []
        self.seq_onehot = []
        self.protein_length = []
        self.pdb_names = []

        for i, row in tqdm(self.csv.iterrows(), total=len(self.csv), desc="load eq structures for training"):
            name = row["name"]
            protein_folder = row["folder"]

            trans, rotmats, seq_onehot, _ = frames_from_pdb(
                os.path.join(protein_folder, f"{name}.pdb")
            )
            seq = np.array([restypes[i] for i in seq_onehot.argmax(1)])

            self.trans_equilibrium.append(trans)
            self.rotmats_equilibrium.append(rotmats)
            self.seq_1letter.append(seq)
            self.seq_onehot.append(seq_onehot)
            self.protein_length.append(len(seq_onehot))
            self.pdb_names.append(name)
            


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Sample data example.
        if isinstance(idx, int):
            pdb_idx = idx
            trajectory_index = 0
            conformation_idx = 0
        else:
            pdb_idx, trajectory_index, conformation_idx = idx

        # mdtraj.load_frame(path, i, top=f"{path[:-7]}.pdb")
        folder = self.csv.iloc[pdb_idx]["folder"]
        name = self.csv.iloc[pdb_idx]["name"]
        traj_path = os.path.join(folder, f"{name}_R{trajectory_index+1}.xtc")
        eq_path = os.path.join(folder, f"{name}.pdb")
        
        conformation = mdtraj.load_frame(traj_path, conformation_idx, top=eq_path)
        N_atoms = conformation.xyz[0, conformation.top.select('name N'), :] * 10
        CA_atoms = conformation.xyz[0, conformation.top.select('name CA'), :] * 10
        C_atoms = conformation.xyz[0, conformation.top.select('name C'), :] * 10

        seq = self.seq_1letter[pdb_idx]

        data = backbone_to_frames(N_atoms, CA_atoms, C_atoms, seq)

        N = self.protein_length[pdb_idx]
        in_feats = {
            "res_mask": torch.ones(N),
            "trans_equilibrium": self.trans_equilibrium[pdb_idx],
            "rotmats_equilibrium": self.rotmats_equilibrium[pdb_idx],
            "trans_1": data["trans"],
            "rotmats_1": data["rotmats"],
            "seq": self.seq_onehot[pdb_idx],
            "pdb_name": self.pdb_names[pdb_idx],
            "conformation_idx": conformation_idx,
            "trajectory_idx": trajectory_index,
        }

        return in_feats

    

class PDBDatasetBBFlowFromPdb(Dataset):
    """
    Used for inference where a csv file with pdb paths
    is given and for the validation set during training.
    """
    def __init__(self, csv, sort=False):
        self.csv = csv

        self.trans = []
        self.rotmats = []
        self.pdb_names = []
        self.seqs = []
        self._preprocess_pdb(sort)

    def _preprocess_pdb(self, sort):
        for i, row in self.csv.iterrows():
            pdb_path = row['pdb_path']
            pdb_name = row['pdb_name']

            trans, rotmats, seq_onehot, _ = frames_from_pdb(pdb_path)

            self.trans.append(trans)
            self.rotmats.append(rotmats)
            self.pdb_names.append(pdb_name)
            self.seqs.append(seq_onehot)

        lengths = [len(seq) for seq in self.seqs]
        if sort:
            indices = np.argsort(lengths)
            self.trans = [self.trans[i] for i in indices]
            self.rotmats = [self.rotmats[i] for i in indices]
            self.pdb_names = [self.pdb_names[i] for i in indices]
            self.seqs = [self.seqs[i] for i in indices]
            self.csv = self.csv.iloc[indices]
            lengths = [lengths[i] for i in indices]
        self.csv.insert(1, 'modeled_seq_len', lengths)


    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        return {
            "pdb_name": self.pdb_names[idx],
            "seq": self.seqs[idx],
            "trans_equilibrium": self.trans[idx],
            "rotmats_equilibrium": self.rotmats[idx],
        }



class LengthBatcherBBFlow:

    def __init__(
            self,
            *,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        self.max_conformations = self._sampler_cfg.max_conformations
        num_total = 0
        for i in range(len(self._data_csv)):
            num_total += min(
                self._data_csv.iloc[i]['num_conformations'], 
                self.max_conformations
            )
        # self._num_batches = math.ceil(num_total / self.num_replicas)
        self._data_csv['index'] = list(range(len(self._data_csv)))
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size = self._sampler_cfg.max_batch_size
        self.max_num_res_squared = self._sampler_cfg.max_num_res_squared
        self._log.info(f'Created dataloader rank {self.rank} out of {self.num_replicas}')
        self.sample_order = None
        lengths = []
        for r in range(self.num_replicas):
            lengths.append(len(self._replica_epoch_batches(r)))
            self._log.info(f"Rank {r} has {lengths[-1]} batches")
        self._num_batches = max(lengths)
        self._log.info(f"Using {self._num_batches} batches per rank")

 
    def _replica_epoch_batches(self, rank=0):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self._data_csv), generator=rng).tolist()
        else:
            indices = list(range(len(self._data_csv)))

        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[
                indices[rank::self.num_replicas]
            ]
        else:
            replica_csv = self._data_csv
        
        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby('modeled_seq_len'):
            max_batch_size = min(
                self.max_batch_size,
                self.max_num_res_squared // seq_len**2 + 1,
            )
            
            protein_indices = []
            for i, row in len_df.iterrows():
                max_conformations = min(
                    self.max_conformations,
                    row['num_conformations']
                )

                # Number of trajectories per protein hardcoded to 3 for now (as in ATLAS)
                traj_indices = np.sort(np.random.choice(3, max_conformations, replace=True))
                traj_indices = torch.tensor(traj_indices)
                conf_indices = []
                for i in range(3):
                    conf_indices.append(np.random.choice(row['num_conformations'], (traj_indices==i).sum().item(), replace=False))
                conf_indices = torch.tensor(np.concatenate(conf_indices))

                protein_index = torch.ones(max_conformations, dtype=torch.long) * row["index"]
                protein_indices.append(torch.stack([
                    protein_index,  # index of protein
                    traj_indices,   # index of trajectory
                    conf_indices,   # index of conformation in trajectory
                ], dim=1))


            protein_indices = torch.cat(protein_indices, dim=0)
            perm = torch.randperm(len(protein_indices), generator=rng)
            protein_indices = protein_indices[perm].numpy()

            num_batches = math.ceil(len(protein_indices) / max_batch_size)
            for i in range(num_batches):
                batch_indices = protein_indices[i*max_batch_size:(i+1)*max_batch_size]
                sample_order.append(batch_indices)
        
        # Remove any length bias.
        new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
        return [sample_order[i] for i in new_order]


    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches(self.rank))
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches
        self._log.info(f"Create batches: Rank {self.rank} has {len(all_batches)} batches.")
        return all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return self._num_batches
        # return len(self.sample_order)
