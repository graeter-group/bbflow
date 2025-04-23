# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch
import numpy as np
import mdtraj

from openfold.data.data_transforms import atom37_to_frames
from openfold.utils.rigid_utils import Rigid

from gafl.data.residue_constants import restype_order


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

