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
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    # residue_index = np.arange(n)
    residue_index = np.arange(n) + 1 # residue index should start from 1 (at least for analyse_bbflow necessary)
    chain_index = np.zeros(n)
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

