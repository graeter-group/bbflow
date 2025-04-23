# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# TODO: Differs from gafl.analysis.utils only in create_full_prot where
#       residue_index is set to start from 1 instead of 0


import numpy as np
import os
import re
from Bio import PDB

from openfold.utils import rigid_utils
from gafl.data import protein
from gafl.analysis.utils import write_prot_to_pdb, get_pdb_length


Rigid = rigid_utils.Rigid


def create_full_prot(
        atom37: np.ndarray,
        atom37_mask: np.ndarray,
        aatype=None,
        b_factors=None,
        chain_index=None,
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    # residue_index = np.arange(n)
    if chain_index is None:
        chain_index = np.zeros(n)
        residue_index = np.arange(n) + 1 # residue index should start from 1 (at least for analyse_bbflow necessary)
    else:
        _, counts = np.unique(chain_index, return_counts=True)
        residue_index = np.concatenate([(np.arange(count) + 1) for count in counts])

    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors)


def write_prot_to_pdb(
        prot_pos: np.ndarray,
        file_path: str,
        aatype: np.ndarray=None,
        overwrite=False,
        no_indexing=False,
        b_factors=None,
        chain_index=None,
    ):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
    if not no_indexing:
        save_path = file_path.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
    else:
        save_path = file_path
    with open(save_path, 'w') as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos37, atom37_mask, aatype=aatype, b_factors=b_factors, chain_index=chain_index)
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos, atom37_mask, aatype=aatype, b_factors=b_factors)
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f'Invalid positions shape {prot_pos.shape}')
        f.write('END')
    return save_path

def get_pdb_length(path: str):
    # Initialize the PDB parser
    parser = PDB.PDBParser(QUIET=True)  # QUIET=True suppresses some warnings

    # Parse the PDB file
    structure = parser.get_structure("sample", path)

    # Initialize a set to track unique residues
    unique_residues = set()

    # Iterate through the chains and residues in the structure
    for model in structure:
        for chain in model:
            for residue in chain:
                # Residue's ID is a tuple (hetero-flag, sequence number, insertion code)
                # Only consider non-hetero residues (ignore water, ions, etc.)
                if residue.id[0] == " ":
                    unique_residues.add(residue.id)

    # Get the total number of unique residues
    return len(unique_residues)