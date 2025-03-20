# %%
import pandas as pd
import numpy as np
import os
import sys
import requests
import re
import json
import os
from tqdm import tqdm
import shutil
import subprocess


class DownloadATLAS():

    def __init__(self,
                ATLAS_csv:str,
                folder_out:str):
        '''
        Args: 
            ATLAS_csv: path to the ATLAS.csv file
            folder_out: output folder where the data will be stored
        '''
        self.df = pd.read_csv(ATLAS_csv)
        self.rmsf_atlas_sorted = self.df.sort_values('Avg.\xa0RMSF', ascending=False)
        self.folder_out = folder_out
        if not os.path.exists(self.folder_out):
            os.makedirs(self.folder_out)

    def _subset_rmsf(self, rmsf_cutoff:int=2):
        '''
        By default filtering by rmsf <= 2 to get rid of too flexible and NMR-derived structures
        '''
        subset_df = self.rmsf_atlas_sorted[self.rmsf_atlas_sorted['Avg.\xa0RMSF']<=rmsf_cutoff]
        self.subset_df = subset_df
        return subset_df

    def _get_pdb_names(self, subset_df):
        pdb_entries = list(subset_df['PDB'])
        self.pdb_entries = pdb_entries
        return pdb_entries

    def _get_name_out(self, pdb):
        uppercase_pattern = r'[A-Z][^A-Z]*'
        identifier = re.findall(uppercase_pattern, pdb)
        split_pdb = re.split(uppercase_pattern, pdb)
        name_out = split_pdb[0]+'_'+identifier[0]
        return name_out

    def _get_pdb_folders(self):
        pdb_folders = [os.path.join(self.folder_out, i) for i in os.listdir(self.folder_out) if os.path.isdir(os.path.join(self.folder_out, i))]
        return pdb_folders

    def _download_data(self, pdb_entries):
        
        '''
        Downloads the .zip files from the ATLAS database and extracts the .xtc and .pdb files
        Args: 
            pdb_entries: list of pdb entries (names)
        '''

        pattern_data = r'\{begin:(\d+),\s+value:(\d+\.\d+),\s+featureId:"value:\s+(\d+\.\d+).'
        pattern_rmsf = r'pfv\.updateTrackData\("rmsfTrack",\[([^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*)\]'
        pattern_plddt = r'rowTitle: "AF2 pLDDT",\s+fitTitleWidth: true,\s+trackData: (\[[^\]]+\])'

        plddt_vals = []
        rmsf_vals = []
        plddt_per_protein = []
        rmsf_per_protein = []

        for pdb_name in tqdm(pdb_entries):
            pdb_out = os.path.join(self.folder_out, pdb_name)
            if not os.path.exists(pdb_out):
                os.makedirs(pdb_out)
            else:
                pass
            correct_name = self._get_name_out(pdb_name)
            results_link = 'https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/'+correct_name+'/'+correct_name+'_analysis.zip'
            cmd = 'wget -q -P {} {}'.format(pdb_out, results_link)

            try:
                subprocess.run(cmd, check=True, stdout=None, stderr=None, shell=True)
            except KeyboardInterrupt:
                print("Interrupted by user, exiting...")
                out_dir = os.path.join(self.folder_out, pdb_name)
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                break
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while executing: {e.cmd}")
                break

            pdb_html = 'https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/'+correct_name+'/'+correct_name+'.html'
            response = requests.get(pdb_html)
            text_scrap = response.text

            plddt_search = re.search(pattern_plddt, text_scrap, re.DOTALL)
            if plddt_search:
                track_plddt = plddt_search.group(1)
            find_plddt = re.findall(pattern_data, track_plddt)
            plddt_vals.append(find_plddt)
            plddt_per_protein.append(find_plddt)
            
            with open(os.path.join(pdb_out,correct_name+'_plddt.json'), 'w') as f:
                json.dump(plddt_per_protein, f)

            rmsf_search = re.search(pattern_rmsf, text_scrap, re.DOTALL)
            if rmsf_search:
                track_rmsf = rmsf_search.group(1)
            find_rmsf = re.findall(pattern_data, track_rmsf)
            rmsf_vals.append(find_rmsf)
            rmsf_per_protein.append(find_rmsf)
        
            with open(os.path.join(pdb_out, correct_name+'_rmsf.json'), 'w') as k:
                json.dump(rmsf_per_protein, k)
        
    def _download_str_rcsb(self, pdb_entries):
        '''
        For each of the ATLAS-deposited selected models downloads the corresponding .pdb file from the RCSB 
        and generates metadata for the downloaded files.
        Args: 
            pdb_entries: list of pdb entries (names)
        '''
        uppercase_pattern = r'[A-Z][^A-Z]*'
        link = 'https://files.rcsb.org/download/'
        metadata_pdbs = {'pdb_name': [],
                        'chain': [],
                        'loc': []}
        pdb_names = [re.split(uppercase_pattern, i)[0] for i in pdb_entries]
        chains = [re.findall(uppercase_pattern, i)[0] for i in pdb_entries]
        
        for index in tqdm(range(len(pdb_names))):
            try:
                cmd = 'wget -q -P {} {}'.format(os.path.join(self.folder_out, pdb_names[index]), link + pdb_names[index] + '.pdb.gz')
                subprocess.run(cmd, check=True, stdout=None, stderr=None, shell=True)
                metadata_pdbs['pdb_name'].append(pdb_names[index])
                metadata_pdbs['chain'].append(chains[index])
                metadata_pdbs['loc'].append(os.path.join(self.folder_out, pdb_names[index], pdb_names[index] + '.pdb.gz'))
            except KeyboardInterrupt:
                print("Interrupted by user, exiting...")
                out_dir = os.path.join(self.folder_out, pdb_names[index])
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                break
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while executing: {e.cmd}")
                break

        metadata_df = pd.DataFrame(metadata_pdbs)
        metadata_df.to_csv(os.path.join(self.folder_out, 'metadata_for_rcsb.csv'))


# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ATLAS_csv', type=str, required=True)
    parser.add_argument('--raw_folder', type=str, required=True)
    parser.add_argument('--rmsf_cutoff', type=float, default=float('inf'))
    parser.add_argument('--subset', type=int, default=0)
    args = parser.parse_args()

    download_samples = DownloadATLAS(ATLAS_csv=args.ATLAS_csv, folder_out=args.raw_folder)
    subset_df = download_samples._subset_rmsf(rmsf_cutoff=args.rmsf_cutoff)
    pdb_entries = download_samples._get_pdb_names(subset_df)
    if args.subset > 0:
        pdb_entries = pdb_entries[:args.subset]
    download_samples._download_data(pdb_entries)
