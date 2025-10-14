# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch
import numpy as np
import mdtraj
import os
from tqdm.auto import tqdm

from openfold.data.data_transforms import atom37_to_frames
from openfold.utils.rigid_utils import Rigid

from gafl.data.residue_constants import restype_order

# from bbflow.analysis.analyse_bbflow import calc_aligned_rmsf
from bbflow.analysis.analyse_bbflow import _calc_rmsf, _align_tops
import json

def backbone_to_frames(N_atoms, CA_atoms, C_atoms, resnames):
    seq_numerical = torch.tensor([restype_order[aa] for aa in resnames])
    seq_onehot = torch.nn.functional.one_hot(seq_numerical, 21).float()

    N = len(N_atoms)
    X = torch.zeros(N, 37, 3)
    X[:, 0] = torch.tensor(N_atoms)
    X[:, 1] = torch.tensor(CA_atoms)
    X[:, 2] = torch.tensor(C_atoms)
    X -= torch.mean(torch.tensor(CA_atoms), dim=0)
    aatypes = torch.tensor([restype_order[aa] for aa in resnames]).long()
    atom_mask = torch.zeros((N, 37)).double()
    atom_mask[:, :3] = 1
    protein = {
        "aatype": aatypes,
        "all_atom_positions": X,
        "all_atom_mask": atom_mask,
    }
    frames = atom37_to_frames(protein)
    rigids_0 = Rigid.from_tensor_4x4(frames['rigidgroups_gt_frames'])[:,0]
    trans = rigids_0.get_trans()
    rotmats = rigids_0.get_rots().get_rot_mats()

    return {
        "trans": trans,
        "rotmats": rotmats,
        "seq_onehot": seq_onehot,
    }

def frames_from_pdb(pdb_path:Path)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts frames from a PDB file.

    Args:
        pdb_path: Path to the PDB file.

    Returns:
        Tuple of trans, rotmats, seq_onehot.
    """

    eq = mdtraj.load(pdb_path)

    N_atoms = eq.xyz[0, eq.top.select('name N'), :] * 10
    CA_atoms = eq.xyz[0, eq.top.select('name CA'), :] * 10
    C_atoms = eq.xyz[0, eq.top.select('name C'), :] * 10
    seq = np.array(list("".join(eq.top.to_fasta())))

    chain_ids = np.array([res.chain.index for res in eq.top.residues])

    data = backbone_to_frames(N_atoms, CA_atoms, C_atoms, seq)
    return data["trans"], data["rotmats"], data["seq_onehot"], chain_ids




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