
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import MDAnalysis as mda
from data.residue_constants import restype_3to1, restype_order
from data.all_atom import transrot_to_atom37, atom37_from_trans_rot
from openfold.data.data_transforms import atom37_to_frames
from openfold.utils.rigid_utils import Rigid
from analysis.utils import write_prot_to_pdb
import zipfile
import biotite.structure.io as strucio
import os

def create_ATLAS():
    conformation_folder = "./dataset/flex_data_window_7/"
    ATLAS_raw_folder = "./dataset/ATLAS_unprocessed/"
    output_folder = "./dataset/ATLAS/"
    data_paths = [f for f in os.listdir(conformation_folder) if f.endswith(".npz")]

    pdb_names = [f.replace(".npz", "").replace("_data_window_7", "").replace("_data", "") for f in data_paths]
    pdb_names_chain = [tuple(f.split("_")) for f in pdb_names]

    metadata = []
    for (pdb_name, chain), data_path in zip(pdb_names_chain, data_paths):
        print(pdb_name, chain, data_path)

        # zip_path = "./dataset/ATLAS_unprocessed/1a62A/1a62_A_analysis.zip"
        zip_path = os.path.join(ATLAS_raw_folder, f"{pdb_name}{chain}/{pdb_name}_{chain}_analysis.zip")

        # pdb_path = "./dataset/ATLAS_unprocessed/1a62A/"
        pdb_path = os.path.join(ATLAS_raw_folder, f"{pdb_name}{chain}/")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            pdb_files = [i for i in all_files if i.endswith('.pdb')]
            pdb_extract = pdb_files[0]
            zip_ref.extract(pdb_extract, pdb_path)

        # u = mda.Universe("./dataset/ATLAS_unprocessed/1a62A/1a62_A.pdb")
        u = mda.Universe(os.path.join(ATLAS_raw_folder, f"{pdb_name}{chain}/{pdb_name}_{chain}.pdb"))
        N_atoms_equilibrium = u.select_atoms("name N").positions
        CA_atoms_equilibrium = u.select_atoms("name CA").positions
        C_atoms_equilibrium = u.select_atoms("name C").positions
        seq = "".join([restype_3to1[aa] for aa in u.residues.resnames])

        # a = np.load("./dataset/flex_data_window_7/1a62_A_data.npz")
        a = np.load(os.path.join(conformation_folder, data_path))

        coords = a["coords"]
        N_atoms_conformation = coords[:, a["atom_name"]=="N"]
        CA_atoms_conformation = coords[:, a["atom_name"]=="CA"]
        C_atoms_conformation = coords[:, a["atom_name"]=="C"]

        B, N = N_atoms_conformation.shape[:2]
        X = torch.zeros(B+1, N, 37, 3)
        X[0, :, 0] = torch.tensor(N_atoms_equilibrium)
        X[0, :, 1] = torch.tensor(CA_atoms_equilibrium)
        X[0, :, 2] = torch.tensor(C_atoms_equilibrium)
        X[1:, :, 0] = torch.tensor(N_atoms_conformation)
        X[1:, :, 1] = torch.tensor(CA_atoms_conformation)
        X[1:, :, 2] = torch.tensor(C_atoms_conformation)
        aatypes = torch.tensor([restype_order[aa] for aa in seq]).long()
        atom_mask = torch.zeros((N, 37)).double()
        atom_mask[:, :3] = 1
        trans = torch.zeros(B+1, N, 3)
        rotmats = torch.zeros(B+1, N, 3, 3)
        for b in range(B+1):
            protein = {
                "aatype": aatypes,
                "all_atom_positions": X[b],
                "all_atom_mask": atom_mask,
            }
            frames = atom37_to_frames(protein)
            rigids_0 = Rigid.from_tensor_4x4(frames['rigidgroups_gt_frames'])[:,0]
            trans[b] = rigids_0.get_trans()
            rotmats[b] = rigids_0.get_rots().get_rot_mats()

        trans.shape

        trans_equilibrium = trans[0]
        trans_conformations = trans[1:]
        rotmats_equilibrium = rotmats[0]
        rotmats_conformations = rotmats[1:]

        data = {
            "pdb_name": pdb_name,
            "seq": np.array([aa for aa in seq]),
            "trans_equilibrium": trans_equilibrium,
            "rotmats_equilibrium": rotmats_equilibrium,
            "trans_conformations": trans_conformations,
            "rotmats_conformations": rotmats_conformations,
        }
        output_path = os.path.join(output_folder, f"{pdb_name}{chain}.npz")
        np.savez(output_path, **data)

        metadata.append({
            "pdb_name": f"{pdb_name}{chain}",
            "modeled_seq_len": len(seq),
            "num_conformations": B,
            "data_path": os.path.abspath(output_path),
        })

    metadata = pd.DataFrame(metadata)
    metadata = metadata.sort_values("modeled_seq_len", ascending=True).reset_index(drop=True)
    metadata.to_csv(os.path.join(output_folder, "metadata.csv"), index=False)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Get filtered ATLAS dataset for flexibility regressor')
    # parser.add_argument('--folder_out', type=str, help='Path to the download folder')
    # parser.add_argument('--batch_number', type=int, help='Batch number for download and local_flex calculation')
    # parser.add_argument('--window_size', type=int, help='Window size for the local flexibility calculation')
    # parser.add_argument('--number_confs', type=int, help='Number of conformations to pick from the trajectory to compute local flexibility')
    # args = parser.parse_args()
    # main(args)
    # print("yo")
    create_ATLAS()