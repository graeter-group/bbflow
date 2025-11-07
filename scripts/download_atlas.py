
import os
import argparse
import shutil
import subprocess
import zipfile
import pandas as pd
from tqdm import tqdm
import mdtraj

def download_data(
    out_root: str,
    train_csv: str=None,
    validation_csv: str=None,
    test_csv: str=None,
    out_csv_folder: str="./dataset_csv",
    max_entries_per_csv: int=None
):
    csv_files = [train_csv, validation_csv, test_csv]
    out_names = ['train', 'validation', 'test']

    for csv_file, out_name in zip(csv_files, out_names):
        if csv_file is None:
            continue
        print(f"Processing CSV file: {csv_file}")
        df_metadata = pd.read_csv(csv_file)

        names = []
        modeled_seq_lens = []
        folders = []
        num_conformations = []
        pdb_paths = []
        
        pbar = tqdm(
            enumerate(df_metadata.iterrows()),
            total=len(df_metadata) if max_entries_per_csv is None else min(len(df_metadata), max_entries_per_csv),
        )
        for i, (index, row) in pbar:
            pdb_name = row['pdb_name'] if 'pdb_name' in row else row['name']
            if "_" in pdb_name:
                name = pdb_name.split("_")[0]
                chain = pdb_name.split("_")[1]
            else:
                name = pdb_name[:4]
                chain = pdb_name[4]
            pdb_name = name + chain
            pbar.set_description(f"Downloading {pdb_name}")
            
            pdb_out_dir = os.path.join(out_root, pdb_name)
            if not os.path.exists(pdb_out_dir):
                os.makedirs(pdb_out_dir)
            else:
                pass

            # Download the analysis zip file, which contains the .pdb and .xtc files
            results_link = f"https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/{name}_{chain}"
            zip_path = os.path.join(pdb_out_dir, f"{pdb_name}_analysis.zip")
            cmd = f"wget -q {results_link} -O {zip_path}"
            try:
                subprocess.run(cmd, check=True, stdout=None, stderr=None, shell=True)
            except KeyboardInterrupt:
                print("Interrupted by user, exiting...")
                if os.path.exists(pdb_out_dir):
                    shutil.rmtree(pdb_out_dir)
                return
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while executing: {e.cmd}")
                break
            
            # Extract .pdb and .xtc files from the zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                xtc_files = [i for i in all_files if i.endswith('.xtc')]
                for xtc in xtc_files:
                    zip_ref.extract(xtc, pdb_out_dir)
                    os.rename(
                        os.path.join(pdb_out_dir, xtc),
                        os.path.join(pdb_out_dir, xtc.replace(f"{name}_{chain}", pdb_name))
                    )
                pdb_files = [i for i in all_files if i.endswith('.pdb')]
                for pdb in pdb_files:
                    zip_ref.extract(pdb, pdb_out_dir)
                    os.rename(
                        os.path.join(pdb_out_dir, pdb),
                        os.path.join(pdb_out_dir, pdb.replace(f"{name}_{chain}", pdb_name))
                    )

            # Remove the zip file after extraction
            if os.path.exists(zip_path):
                os.remove(zip_path)

            # Collect metadata
            names.append(pdb_name)
            pdb_paths.append(os.path.abspath(os.path.join(pdb_out_dir, f"{pdb_name}.pdb")))
            if out_name == 'train':
                folders.append(os.path.abspath(pdb_out_dir))
                if 'modeled_seq_len' in row and 'num_conformations' in row:
                    modeled_seq_lens.append(row['modeled_seq_len'])
                    num_conformations.append(row['num_conformations'])
                else:
                    traj = mdtraj.load(
                        os.path.join(pdb_out_dir, xtc.replace(f"{name}_{chain}", pdb_name)),
                        top=os.path.join(pdb_out_dir, f"{pdb_name}.pdb")
                    )
                    modeled_seq_lens.append(traj.n_residues)
                    num_conformations.append(traj.n_frames)


            if max_entries_per_csv is not None and i + 1 >= max_entries_per_csv:
                break

        out_metadata = {
            'name': names,
        }
        if out_name == 'train':
            out_metadata['modeled_seq_len'] = modeled_seq_lens
            out_metadata['pdb_path'] = pdb_paths
            out_metadata['folder'] = folders
            out_metadata['num_conformations'] = num_conformations
        else:
            out_metadata['pdb_path'] = pdb_paths

        out_df = pd.DataFrame(out_metadata)
        if out_name == 'train':
            out_df = out_df.sort_values(by='modeled_seq_len').reset_index(drop=True)
        
        out_csv_folder_ = os.path.join(out_csv_folder, out_name)
        if not os.path.exists(out_csv_folder_):
            os.makedirs(out_csv_folder_)
        out_csv_path = os.path.join(out_csv_folder_, f"ATLAS_{out_name}.csv")
        out_df.to_csv(out_csv_path, index=False)
        print(f"Saved metadata to {out_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download ATLAS dataset')
    parser.add_argument('--out_folder', type=str, default='./datasets/ATLAS', help='Path to the download folder')
    parser.add_argument('--train_csv', type=str, default='dataset_csv/ATLAS_split/ATLAS_train.csv', help='Path to the training CSV file with PDB entries to download')
    parser.add_argument('--validation_csv', type=str, default='dataset_csv/ATLAS_split/ATLAS_validation.csv', help='Path to the validation CSV file with PDB entries to download')
    parser.add_argument('--test_csv', type=str, default='dataset_csv/ATLAS_split/ATLAS_test.csv', help='Path to the test CSV file with PDB entries to download')
    parser.add_argument('--out_csv_folder', type=str, default="./dataset_csv", help='Folder to save the output CSV files')
    parser.add_argument('--max_entries_per_split', type=int, default=None, help='Maximum number of entries to download per CSV file')
    args = parser.parse_args()

    # Example usage:
    # python scripts/download_atlas.py 
    #       --out_folder=./datasets/ATLAS
    #       --train_csv=./dataset_csv/ATLAS_split/ATLAS_train.csv
    #       --validation_csv=./dataset_csv/ATLAS_split/ATLAS_validation.csv
    #       --test_csv=./dataset_csv/ATLAS_split/ATLAS_test.csv
    #       --out_csv_folder=./dataset_csv/


    download_data(
        out_root=args.out_folder,
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        test_csv=args.test_csv,
        out_csv_folder=args.out_csv_folder,
        max_entries_per_csv=args.max_entries_per_split
    )




