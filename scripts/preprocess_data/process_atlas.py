import pandas as pd
import numpy as np
import re
import json
import os
from tqdm import tqdm
import zipfile
import MDAnalysis as mda
import gzip
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdb as pdb
import biotite.structure.superimpose as superimpose
import biotite.structure.compare as compare
import biotite.structure.atoms as atoms_tite
import logging 
from pathlib import Path
import argparse

class ProcessDataATLAS():

    '''
    Version implemented to compute local flexibility per trajectory instead of pulled per-res flex

    TODO: for 900 conf from the trajectory, compute for 3 randomly picked reference conformations the per-residue rmsd
    TODO: for this per-traj derived local flex, compute means and stds AND store both values before averaging and after averaging in an npz
    TODO: after that pull with two remaining traj-derived data and compute global means and stds
    '''

    def __init__(self,
                  folder_out:str,
                metadata_csv:str=None,
                batch_number:int=None, 
                window_size:int=7,
                number_confs:int=900):

        '''
        Args: 
            batch_number: number of the batch to process (multinode-processing)
            number_confs: number of conformations to pick from the trajectory to compute local flexibility
        '''
        self.folder_out = folder_out
        self.pdb_folders = self._get_input_folders()
        self.number_confs = number_confs
        self.window_size = window_size
        if metadata_csv is not None:
            self.metadata = pd.read_csv(metadata_csv)
        else:
            self.metadata = None
        self.batch_number = batch_number
    
    def _get_input_folders(self):
        input_pdb_folders = [os.path.join(self.folder_out, i) for i in os.listdir(self.folder_out) if os.path.isdir(os.path.join(self.folder_out, i))]
        return input_pdb_folders
    
    def _subset_locs(self):
        '''
        Subsets the pdb_folders based on the batch_number [0, 4] into 5 chunks for multinode-processing
        '''
        if self.metadata is not None:
            total = len(self.metadata)
        else:
            total = len(self.pdb_folders)
        n_parts = 5
        part_size = total // n_parts
        indexes = [i * part_size for i in range(n_parts)] + [total]
        slices = [(indexes[i], indexes[i+1]) for i in range(len(indexes)-1)]
        return slices[self.batch_number]

    def _get_name_out(self, pdb:str):
        uppercase_pattern = r'[A-Z][^A-Z]*'
        identifier = re.findall(uppercase_pattern, pdb)
        split_pdb = re.split(uppercase_pattern, pdb)
        name_out = split_pdb[0]+'_'+identifier[0]
        return name_out
    
    def _get_structure_tite(self, pdb_loc:str, compressed:bool=False):
        if compressed:
            with gzip.open(pdb_loc, "rt") as file_handle:
                pdb_file = pdb.PDBFile.read(file_handle)
        else:
            with open(pdb_loc, "r") as file_handle:
                pdb_file = pdb.PDBFile.read(file_handle)
        protein = pdb.get_structure(pdb_file, model=1, extra_fields=['b_factor'])
        protein = protein[protein.hetero == False]
        return protein
    
    def _rename_chain(self, protein:struc.AtomArray, chain:str='A'):
        '''
        Rename each of the chain_id to 'A'
        '''
        protein.chain_id[:] = chain
        return protein
    
    def _get_sequence(self, protein:struc.AtomArray):
        d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
            'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
            'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
            'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        
        res_names = protein[protein.atom_name == 'CA'].res_name
        res_names = [d[i] for i in res_names]
        sequence = ''.join(res_names)
        return sequence
    
    def _extract_pdb_xtc(self):
        # print('--- Extracting .xtc and .pdb files ---')
        logging.info('--- Extracting .xtc and .pdb files ---')
        for pdb_folder in tqdm(self.pdb_folders):
            pdb_path = os.path.join(self.folder_out, pdb_folder)
            assert os.path.exists(pdb_path) == True, 'Check whether the folder exists'
            pdb_name = self._get_name_out(pdb_folder)
            if os.path.isdir(pdb_path):
                files_list = [i for i in os.listdir(pdb_path) if i.endswith('.zip')]
                try:
                    zip_file = os.path.join(pdb_path, files_list[0])
                except IndexError:
                    # print('No .zip files found in the folder')
                    logging.error(f'No .zip files found in the folder {pdb_folder}')
                    continue
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    all_files = zip_ref.namelist()
                    xtc_files = [i for i in all_files if i.endswith('.xtc')]
                    pdb_files = [i for i in all_files if i.endswith('.pdb')]
                    pdb_extract = pdb_files[0]
                    for xtc in xtc_files:
                        zip_ref.extract(xtc, pdb_path)
                    zip_ref.extract(pdb_extract, pdb_path)
                    
                    pdb_loc = os.path.join(pdb_path, pdb_extract)
                    protein_structure = self._get_structure_tite(pdb_loc)
                    # rename the chain differently by appending the actual chain name from the pdb_name
                    # protein_structure = self._rename_chain(protein_structure)
                    strucio.save_structure(pdb_loc, protein_structure)
                    sequence = self._get_sequence(protein_structure)
                    with open(os.path.join(pdb_path, f'{pdb_name}_sequence.json'), 'w') as f:
                        json.dump(sequence, f)

    def _pick_conformations_per_traj(self, pdb_path:str):

        '''
        From the triplicate of MD simulations, pick 100 random conformations
        Args: 
            pdb_path: path to the folder with .pdb and .xtc files
        Returns: 
            per_traj_data: dictionary with keys 'picked_confs_traj_{index}' and 'coords_traj_{index}'; coords_traj_{index} [N_conf, L_prot, 3, 3]
        '''
                
        def _frames_to_pick(range_start: int, range_end: int, total_picks: int= 900):
            '''
            Args: 
                num_sets: number of sets of frames
                range_start: starting frame
                range_end: ending frame
            '''
            return np.random.choice(range(range_start, range_end), total_picks, replace=False)

        def _get_positions_conf(pdb_iter, run, picked_frames):
            '''
            Returns: 
                frames_list: [n_conf, n_residues, 3, 3]
            '''
            universe = mda.Universe(pdb_iter, run)
            backbone = universe.select_atoms('name N CA C')
            frames_list = []
            for conf, frame in enumerate(universe.trajectory):
                # this might be a problem if conf is not in picked_frames
                if conf in picked_frames:
                    residues_frames = []
                    for i in range(universe.residues.n_residues):
                        # picking only the backbone atoms
                        start, end = i*3, i*3+3
                        residue_frame = backbone.positions[start:end]
                        residues_frames.append(residue_frame)
                    frames_list.append(residues_frames)
            return frames_list
        
        pdb_files = [i for i in os.listdir(pdb_path) if i.endswith('.pdb')]
        xtc_files = [i for i in os.listdir(pdb_path) if i.endswith('.xtc')]
        pdb_iter = os.path.join(pdb_path, pdb_files[0])

        try:
            runs = [os.path.join(pdb_path, xtc_files[i]) for i in range(len(xtc_files))]
        except:
            print('check whether all three .xtc are in the folder!')
            return None
        
        # get the number of frames in the trajectory, assuming lens of trajs are 1000
        universe_iter = mda.Universe(pdb_iter, runs[0])
        num_frames = universe_iter.trajectory.n_frames
        # start from the 100th frame -> equillibrated
        frame_start = 100

        per_traj_data = {}
        for index, run in enumerate(runs):
            # print(f'--- Picking {self.number_confs} conformations from the trajectory {index} ---')
            logging.info(f'--- Picking {self.number_confs} conformations from the trajectory {index} ---')
            picked_confs = _frames_to_pick(range_start=frame_start, range_end=num_frames, total_picks=self.number_confs)
            # for one pdb structure, load the trajectory and pick the conformations
            result_confs = _get_positions_conf(pdb_iter, run, picked_confs)
            per_traj_data[f'picked_confs_traj_{index}'] = picked_confs
            per_traj_data[f'coords_traj_{index}'] = np.array(result_confs)
        return per_traj_data

    def _get_locs_rmsf(self):
        '''
        Generated in the download step
        '''
        for folder in tqdm(self.pdb_folders):
            rmsf_file = [i for i in os.listdir(folder) if i.endswith('rmsf.json')]
            with open(os.path.join(folder, rmsf_file[0]), 'r') as f:
                rmsf_data = json.load(f)
            rmsf_vals = [float(i[2]) for i in rmsf_data[0]]
        return rmsf_vals

    def make_dict_bb(self,
                  list_stacked_confs:list,
                  flexibility:bool=False):
        '''
        stacked_conf: AtomArrayStack with 100 conformations generated by _convert_confs_to_tite()
        Returns:
            dict_store: dict containing the data chain_feats for each of the trajectories [0, 1, ..., n | n = 3]
        '''
        store_chain_feats_per_traj = []
        for index in range(len(list_stacked_confs)):
            stacked_conf = list_stacked_confs[index]
            info = {'coords': [],
                    'chain_id': [],
                    'res_id': [],
                    'res_name': [],
                    'atom_name': [],
                    'element': [],
                    'b_factor': [],
                    'local_flex':[], 
                    'std_local_flex':[]}

            # assuming that every conformation has the same res_id, res_name etc: 
            coords = [i for i in stacked_conf.coord]
            chain_id = stacked_conf.chain_id
            res_id = stacked_conf.res_id
            res_name = stacked_conf.res_name
            atom_name = stacked_conf.atom_name
            element = stacked_conf.element
            b_factor = np.zeros(len(stacked_conf))

            info['coords'] = coords
            info['chain_id'] = chain_id
            info['res_id'] = res_id
            info['res_name'] = res_name
            info['atom_name'] = atom_name
            info['element'] = element
            info['b_factor'] = b_factor
            
            if flexibility is not False:
                assert type(flexibility) == list, 'arg flexibility must be provided as a list containing flexibility data for n trajectories'
                means = flexibility[index]['means_per_res']
                stds = flexibility[index]['std_per_res']
                info['local_flex'] = np.repeat(means, 3)
                info['std_local_flex'] = np.repeat(stds, 3)

            store_chain_feats_per_traj.append(info)
        dict_store = {'chain_feats_per_traj': store_chain_feats_per_traj}
        return dict_store
    
    def _write_npz(self, info_bb, pdb_name):
        np.savez(os.path.join(self.folder_out, f'{pdb_name}.npz'), **info_bb)
    
    def _convert_confs_to_tite(self, 
                            dict_confs: dict):
        
        '''
        Args: 
            dict_conf: dict with keys coords_traj_{0-2} with shape [N_conf, L_residues, 3, 3]
        Returns:
            atoms_tite_list [n_traj (def = 3), N_conf, L_prot]: a list with 3 AtomArrayStack() obj containing n conformations per-trajectory stored as chain_feats
        # '''

        coords_all = [dict_confs['coords_traj_0'], dict_confs['coords_traj_1'], dict_confs['coords_traj_2']]

        # print(f'--- Converting {len(coords_all)} trajectories to AtomArrayStack ---')
        logging.info(f'--- Converting {len(coords_all)} trajectories to AtomArrayStack ---')

        atoms_tite_list = []
        for coord_traj in tqdm(coords_all):
            assert len(coord_traj) == self.number_confs, f'Number of conformations must be {self.number_confs}'
            # reshape the coords to [N_atoms, 3] instead of [N_res, 3, 3]; 
            # TODO: return already the correct frame-wise coords [N_res, 3, 3]
            coords_per_conf = list(map(lambda x: x.reshape(-1, 3), coord_traj))
            shape_reshaped_coords = coord_traj[0].reshape(-1, 3)
            atom_annot = np.array(['N', 'CA', 'C']*int(len(shape_reshaped_coords)/3))

            conformations = []
            for coords in coords_per_conf:
                chain_feats = struc.AtomArray(len(coords))
                chain_feats.coord = coords
                chain_feats.atom_name = atom_annot
                chain_feats.chain_id[:] = 'A'
                # mask residue names
                chain_feats.res_name = ['ALA']*int(len(coords))
                chain_feats.res_id = np.arange(1, len(coords)/3+1).repeat(3)
                conformations.append(chain_feats)
            atoms_tite_list.append(atoms_tite.stack(conformations))
        return atoms_tite_list
    
    def _calc_local_flex(self,
                        stacked_confs:atoms_tite.AtomArrayStack,
                        window_size:int,
                        min_spacing_refs:int=None, 
                        num_refs:int=3):
        '''
        Returns: 
            list of shape [num_refs, num_confs-1, N_residues]
        '''

        def pick_indices(num_confs, min_spacing_refs, num_refs=num_refs):    
            while True:
                indices = np.random.choice(num_confs, num_refs, replace=False)
                if all(abs(indices[i] - indices[j]) >= min_spacing_refs for i in range(len(indices)) for j in range(i + 1, len(indices))):
                    return indices

        num_confs = len(stacked_confs)
        assert num_confs == self.number_confs, f'Number of conformations must be {self.number_confs}'
        # dynamically define it here
        min_spacing_refs = num_confs // 10
        random_indices = pick_indices(num_confs, min_spacing_refs, num_refs=3)
        ref_confs = [stacked_confs[i] for i in random_indices]

        half_window = window_size // 2
        rmsds_per_reference = []

        # print(f'--- Calculating rmsd for {len(ref_confs)} reference conformations ---')
        logging.info(f'--- Calculating rmsd for {len(ref_confs)} reference conformations ---')
        for index in tqdm(range(len(ref_confs))):
            ref_CA = ref_confs[index][ref_confs[index].atom_name == "CA"]
            rmsd_per_conf = []

            compare_confs = [stacked_confs[j] for j in range(num_confs) if j != random_indices[index]]
            for conformation in compare_confs:
                conf_CA = conformation[conformation.atom_name == "CA"]
                rmsd_per_res = []
                for i in range(len(conf_CA)):
                    start_index = start_index = max(0, i - half_window)
                    end_index = min(len(conformation), i + half_window + 1)

                    window_values_conf = conf_CA[start_index:end_index]
                    window_values_ref = ref_CA[start_index:end_index]

                    array2_fit, transformation = superimpose(
                    window_values_ref, window_values_conf, atom_mask=(window_values_ref.atom_name == "CA"))

                    # computing rmsd only for a single residue based on the superposition of the whole window
                    rmsd = compare.rmsd(ref_CA[ref_CA.res_id == i+1], array2_fit[array2_fit.res_id == i+1])        
                    rmsd_per_res.append(rmsd)
                rmsd_per_conf.append(rmsd_per_res)
            rmsds_per_reference.append(np.array(rmsd_per_conf))
        return rmsds_per_reference

    def _get_loc_flex(self, list_traj_stacked_confs:list, folder:str, window_size:int=7):
        '''
        For each of the <3> trajectories, compute mean per-residue local flexibility and std
        Args:
            list_traj_stacked_confs: list of 3 AtomArrayStack objects with 899 conformations
            folder: path to the folder where the npz file will be stored
        Returns: 
            dict_store_flex: dict with keys 'flex_per_traj' and 'dict_global_flex'; flex_per_traj: list of dicts with keys 'per_res_flex' (len=3 bc 3 refs), 'means_per_res', 'std_per_res'
        '''
        # try:
        store_flex_per_traj = []
        # print(f'--- Computing local flexibility for {len(list_traj_stacked_confs)} trajectories ---')
        logging.info(f'--- Computing local flexibility for {len(list_traj_stacked_confs)} trajectories ---')
        for traj_data in tqdm(list_traj_stacked_confs):
            all_ref_flex = self._calc_local_flex(traj_data, window_size=window_size)
        # except: 
        #     logging.error(f'Error in computing local flexibility in folder: {folder}')
        #     print(f'Error in computing local flexibility in folder: {folder}')
        #     return None

        # for each of the traj compute mean and std, store them in a dictionary
            means_all = []
            dict_flex_out = {}
            for ref_flex in all_ref_flex:
                # what the fuck am I looping through here????
                mean_rmsd = ref_flex.mean(axis=0)
                means_all.append(mean_rmsd)
                # len of means_per_res = N_confs - 1
            means_all = np.array(means_all)
            means_per_res = means_all.mean(axis=0)
            std_per_res = means_all.std(axis=0)
            dict_flex_out = {'per_res_flex': all_ref_flex, 'means_per_res': means_per_res, 'std_per_res': std_per_res}
            store_flex_per_traj.append(dict_flex_out)

        means_per_traj = [i['means_per_res'] for i in store_flex_per_traj]
        means_per_traj = np.array(means_per_traj)
        global_means = means_per_traj.mean(axis=0)
        global_std = means_per_traj.std(axis=0)
        dict_global_flex = {'global_means_per_res': global_means, 'global_std_per_res': global_std}
        dict_store_flex = {'flex_per_traj': store_flex_per_traj, 'dict_global_flex': dict_global_flex}

        correct_name_out = self._get_name_out(folder.split('/')[-1])
        # np.savez_compressed(os.path.join(folder, f'{correct_name_out}_alltraj_loc_flexibility_window_{window_size}'), **dict_global_flex)
        np.savez_compressed(os.path.join(folder, f'{correct_name_out}_alldata_flexibility_window_{window_size}'), **dict_store_flex)
        # np.savez_compressed(os.path.join(folder, f'{correct_name_out}_loc_flexibility_window_{window_size}'), means_per_res=means_per_res, std_per_res=std_per_res)
        return store_flex_per_traj

    def _remove_temp_files(self, folder:str):
        xtc_files = [i for i in os.listdir(folder) if i.endswith('.xtc')]
        zip_files = [i for i in os.listdir(folder) if i.endswith('.zip')]
        for xtc in xtc_files:
            os.remove(os.path.join(folder, xtc))
        for zip in zip_files:
            os.remove(os.path.join(folder, zip))

    def _generate_dataset(self, exctract_xtc:bool=False, clean_up:bool=False):

        '''
        Args:
            exctract_xtc: whether to extract the .xtc and .pdb files from the .zip
            clean_up: whether to remove the .xtc and .zip files after data generation
        Returns: 
            npz files with the flexibility data:
                - {pdb_name}_data_window_{window_size}.npz containing the chain_feats for each of the trajectories
                - {pdb_name}_alldata_flexibility_window_{window_size}.npz containing the local flexibility data for each of the trajectories
        '''

        metadata = {
            'pdb_name':[],
            'chain_id':[],
            'modeled_seq_len':[],
            'processed_path':[]
        }

        if self.batch_number is not None:
            log_file_path = os.path.join(self.folder_out, f'metadata_log{self.batch_number}_window_{self.window_size}.log')
        else: 
            log_file_path = os.path.join(self.folder_out, f'metadata_log_window_{self.window_size}.log')
        logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                      format='%(asctime)s - %(levelname)s - %(message)s')
        
        if exctract_xtc:
            # first extract all data needed from the .tar: 
            self._extract_pdb_xtc()
            logging.info(f'Extracted .xtc and .pdb files')
        
        if self.batch_number is not None:
            start, end = self._subset_locs()
            self.pdb_folders = self.pdb_folders[start:end]
            logging.info(f'Indices: {start}, {end}')
            logging.info(f'Processing {len(self.pdb_folders)} folders with window size {self.window_size}')

        # print(f'Processing {len(self.pdb_folders)} folders with window size {self.window_size}')
        logging.info(f'Processing {len(self.pdb_folders)} folders with window size {self.window_size}')
        for folder in tqdm(self.pdb_folders):
            _folder_out = os.path.join(self.folder_out, folder.split('/')[-1])
            logging.info(f'Processing folder: {folder}')
            # try:
            conformations = self._pick_conformations_per_traj(folder)
            tite_confs = self._convert_confs_to_tite(dict_confs=conformations)
            # except:
                # logging.error(f'Error in processing the folder: {folder}, skipping')
                # continue

            try:
                data_flexibility = self._get_loc_flex(tite_confs, _folder_out, window_size=self.window_size)
            except:
                logging.error(f'Error in computing local flexibility in folder: {folder}')
                # print(f'Error in computing local flexibility in folder: {folder}')
                continue
            dict_tite = self.make_dict_bb(list_stacked_confs=tite_confs, flexibility=data_flexibility)
            if not os.path.exists(_folder_out):
                os.makedirs(_folder_out)

            name_out = self._get_name_out(_folder_out.split('/')[-1])
            np.savez(os.path.join(self.folder_out, folder, f'{name_out}_data_window_{self.window_size}'), **dict_tite)

            metadata['pdb_name'].append(name_out)
            # TODO: chainA is incorrect; append the right one
            metadata['chain_id'].append('A')
            # extracting sequence length from the first trajectory's residues length
            seq_len = tite_confs[0].res_id[::3].shape[0]
            metadata['modeled_seq_len'].append(seq_len)
            metadata['processed_path'].append(os.path.join(self.folder_out, folder, f'{name_out}_alldata_flexibility_window_{self.window_size}.npz'))
            logging.info(f'--- Successfully finished processing folder: {folder} ---')

        metadata_df = pd.DataFrame(metadata)
        if self.batch_number is not None:
            metadata_df.to_csv(os.path.join(self.folder_out, f'metadata_{self.batch_number}_window_{self.window_size}.csv'))
        else:
            metadata_df.to_csv(os.path.join(self.folder_out, f'metadata_window_{self.window_size}.csv'))

        if clean_up:
            # remove the .xtc and .zip files after data generation
            logging.info('--- Removing .xtc and .zip files ---')
            for folder in self.pdb_folders:
                _folder_out = os.path.join(self.folder_out, folder.split('/')[-1])
                self._remove_temp_files(_folder_out)
            logging.info('--- Done ---')

# testing_folder = '/hits/fast/mbm/viliugvd/databases/ATLAS/test_atlas'
# test_process = ProcessDataATLAS(folder_out=testing_folder, number_confs=100)
# folders = test_process._get_input_folders()
# xxx = test_process._generate_dataset(window_size=7, exctract_xtc=True, clean_up=False)
# test_npz = np.load('/hits/fast/mbm/viliugvd/databases/ATLAS/test_atlas/2f6eA/2f6e_A_alldata_flexibility_window_7.npz', allow_pickle=True)
# test_npz then contains all the data: per-res flexibility per traj and then global stuff as well. 

def main(args):
    folder_out = args.folder_out
    batch_number = args.batch_number
    window_size = args.window_size
    number_confs = args.number_confs
    test_process = ProcessDataATLAS(folder_out=folder_out, batch_number=batch_number, window_size=window_size, number_confs=number_confs)
    test_process._generate_dataset(exctract_xtc=False, clean_up=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get filtered ATLAS dataset for flexibility regressor')
    parser.add_argument('--folder_out', type=str, help='Path to the download folder')
    parser.add_argument('--batch_number', type=int, help='Batch number for download and local_flex calculation')
    parser.add_argument('--window_size', type=int, help='Window size for the local flexibility calculation')
    parser.add_argument('--number_confs', type=int, help='Number of conformations to pick from the trajectory to compute local flexibility')
    args = parser.parse_args()
    main(args)