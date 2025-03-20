
import zipfile
import os
import shutil

def extract_atlas_xtc(raw_folder, out_folder):

    folders = [f for f in os.listdir(raw_folder) if os.path.isdir(os.path.join(raw_folder, f))]
    paths = [os.path.join(raw_folder, f) for f in folders]
    pdb_names_chain = [(f[:4], f[4]) for f in folders]

    for (pdb_name, chain), data_path in zip(pdb_names_chain, paths):
        print(pdb_name, chain, data_path)

        # zip_path = "./dataset/ATLAS_unprocessed/1a62A/1a62_A_analysis.zip"
        zip_path = os.path.join(data_path, f"{pdb_name}_{chain}_analysis.zip")

        # pdb_path = "./dataset/ATLAS_unprocessed/1a62A/"
        pdb_path = os.path.join(raw_folder, f"{pdb_name}{chain}/")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            xtc_files = [i for i in all_files if i.endswith('.xtc')]
            for xtc in xtc_files:
                zip_ref.extract(xtc, pdb_path)

        new_xtc_path = os.path.join(out_folder, f"{pdb_name}{chain}")
        os.makedirs(new_xtc_path, exist_ok=True)
        for xtc in xtc_files:
            shutil.copy(
                os.path.join(data_path, xtc),
                os.path.join(new_xtc_path, f"{pdb_name}{chain}_{xtc[-6:]}")
            )
        shutil.copy(
            os.path.join(data_path, f"{pdb_name}_{chain}.pdb"),
            os.path.join(new_xtc_path, f"{pdb_name}{chain}.pdb")
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract ATLAS xtc files')
    parser.add_argument('--raw_folder', type=str, help='Path to the raw ATLAS data folder')
    parser.add_argument('--out_folder', type=str, help='Path to the output folder')
    args = parser.parse_args()
    extract_atlas_xtc(args.raw_folder, args.out_folder)
