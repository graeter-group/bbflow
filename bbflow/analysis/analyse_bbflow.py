# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import numpy as np
import warnings
import pandas as pd
import scipy.stats
import os
import functools
from sklearn.decomposition import PCA
from scipy.stats import bootstrap
import mdtraj
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linear_sum_assignment
import MDAnalysis as mda
import warnings
import time



def _correlations(a, b, correlation_type="pearson"):
    if a.shape[0] == 1 or b.shape[0] == 1:
        return np.nan
    if correlation_type == "pearson":
        return scipy.stats.pearsonr(a, b)[0]
    elif correlation_type == "spearman":
        return scipy.stats.spearmanr(a, b)[0]
    elif correlation_type == "kendall":
        return scipy.stats.kendalltau(a, b)[0]

def _get_pca(xyz):
    traj_reshaped = xyz.reshape(xyz.shape[0], -1)
    pca = PCA(n_components=min(traj_reshaped.shape))
    coords = pca.fit_transform(traj_reshaped)
    return pca, coords

def _get_rmsds(traj1, traj2, broadcast=False):
    n_atoms = traj1.shape[1]
    traj1 = traj1.reshape(traj1.shape[0], n_atoms * 3)
    traj2 = traj2.reshape(traj2.shape[0], n_atoms * 3)
    if broadcast:
        traj1, traj2 = traj1[:,None], traj2[None]
    distmat = np.square(traj1 - traj2).sum(-1)**0.5 / n_atoms**0.5 * 10
    return distmat

def _get_mean_covar(xyz):
    mean = xyz.mean(0)
    xyz = xyz - mean
    covar = (xyz[...,None] * xyz[...,None,:]).mean(0)
    return mean, covar

def _sqrtm(M):
    D, P = np.linalg.eig(M)
    out = (P * np.sqrt(D[:,None])) @ np.linalg.inv(P)
    return out

def _get_wasserstein(distmat, p=2):
    assert distmat.shape[0] == distmat.shape[1]
    distmat = distmat ** p
    row_ind, col_ind = linear_sum_assignment(distmat)
    return distmat[row_ind, col_ind].mean() ** (1/p)

def _get_names(top, ignore_resname=False):
    names = []
    for residue in top.residues:
        for atom in residue.atoms:
            name = f"{atom.residue.chain.index}-{atom.residue.index}"
            if not ignore_resname:
                name += f"-{atom.residue.name}"
            name += f"-{atom.name}"
            names.append(name)
    return names

def _align_tops(top1, top2, ignore_resname=False):
    names1 = _get_names(top1, ignore_resname)
    names2 = _get_names(top2, ignore_resname)

    intersection = [nam for nam in names1 if nam in names2]
    
    mask1 = [names1.index(nam) for nam in intersection]
    mask2 = [names2.index(nam) for nam in intersection]
    return sorted(mask1), sorted(mask2)

def _calculate_dccm_vectorized(traj):
    ca_indices = traj.topology.select('name CA')
    ca_coords = traj.xyz[:, ca_indices, :]
    
    n_frames = traj.n_frames
    n_ca = len(ca_indices)
    ca_coords_reshaped = ca_coords.reshape(n_frames, 3 * n_ca)
    
    # mean coordinates and displacements for each atom over time
    mean_coords = np.mean(ca_coords_reshaped, axis=0)
    displacements = ca_coords_reshaped - mean_coords
    
    covariance_matrix = np.cov(displacements.T)
    block_covariances = covariance_matrix.reshape(n_ca, 3, n_ca, 3)

    # Calculate the RMSD for each atom by taking the trace of the 3x3 covariance submatrices
    cov = np.einsum('ikjk->ij', block_covariances)
    rmsd = np.sqrt(np.diag(cov))
    cov[cov == 0] = 1e-8
    
    dccm = cov / (rmsd[:, None] * rmsd[None, :])

    return dccm

def _calc_rmsd(ref_traj_1, ref_traj_2, model_traj):
    # print("Calculating RMSD")
    time_start = datetime.now()
    ref_rmsd_matrix = _get_rmsds(ref_traj_1.xyz, ref_traj_2.xyz, broadcast=True)
    model_rmsd_matrix = _get_rmsds(model_traj.xyz, model_traj.xyz, broadcast=True)
    ref_mean_pairwise_rmsd = ref_rmsd_matrix.mean()
    model_mean_pairwise_rmsd = model_rmsd_matrix.mean()
    return {
        'ref_mean_pairwise_rmsd': ref_mean_pairwise_rmsd,
        'model_mean_pairwise_rmsd': model_mean_pairwise_rmsd,
        'ref_rmsd_matrix': ref_rmsd_matrix,
        'model_rmsd_matrix': model_rmsd_matrix,
        'rmsd_time_elapsed': (datetime.now() - time_start).total_seconds(),
    }

def _center_coordinates(traj: mdtraj.Trajectory):
    """
    mdtraj's parallel centering function might cause problems with multiprocessing.
    """
    means = np.mean(traj.xyz.astype(np.float64), axis=1, keepdims=True)
    traj.xyz -= means
    traces = np.sum(traj.xyz**2, axis=(1,2))
    traj._rmsd_traces = traces
    return traj

def _rmsf_bootstrap(trajs, indices):
    # rmsf changes the trajectory inplace, so we need to clone it
    a = mdtraj.Trajectory(trajs[0].xyz.copy(), trajs[0].top)
    b = mdtraj.Trajectory(trajs[1].xyz.copy(), trajs[1].top)
    a = _center_coordinates(a)
    b = _center_coordinates(b)
    return mdtraj.rmsf(a[indices], b, parallel=False, precentered=True) * 10

def _calc_rmsf(traj, ref, bootstrap=False):
    # print("Calculating RMSF")
    if bootstrap:
        rmsf, low, high = _bootstrap(
            _rmsf_bootstrap, [traj, ref], traj.n_frames, n_iterations=100, alpha=0.05
        )
        return {
            'rmsf': rmsf,
            'rmsf_low': low,
            'rmsf_high': high,
        }
    else:
        traj = _center_coordinates(traj)
        ref = _center_coordinates(ref)
        return mdtraj.rmsf(traj, ref, parallel=False, precentered=True) * 10

def _calc_dccm(ref_traj, model_traj):
    # print("Calculating DCCM")
    time_start = datetime.now()
    ref_dccm = _calculate_dccm_vectorized(ref_traj)
    model_dccm = _calculate_dccm_vectorized(model_traj)
    return {
        'ref_dccm': ref_dccm,
        'model_dccm': model_dccm,
        'dccm_time_elapsed': (datetime.now() - time_start).total_seconds(),
    }

def _calc_rmwd(ref_traj, model_traj):
    # print("Calculating RMWD")
    time_start = datetime.now()
    ref_mean, ref_covar = _get_mean_covar(ref_traj.xyz)
    model_mean, model_covar = _get_mean_covar(model_traj.xyz)
    emd_mean = (np.square(ref_mean - model_mean).sum(-1) ** 0.5) * 10
    try:
        emd_var = (np.trace(ref_covar + model_covar - 2*_sqrtm(ref_covar @ model_covar), axis1=1,axis2=2) ** 0.5) * 10
    except:
        emd_var = np.trace(ref_covar) ** 0.5 * 10
    
    rmwd_trans = np.square(emd_mean).mean() ** 0.5
    rmwd_var = np.square(emd_var).mean() ** 0.5
    rmwd = np.sqrt(rmwd_trans ** 2 + rmwd_var ** 2)

    return {
        # 'emd_mean': emd_mean,
        # 'emd_var': emd_var,
        'rmwd': rmwd,
        'rmwd_trans': rmwd_trans,
        'rmwd_var': rmwd_var,
        'rmwd_time_elapsed': (datetime.now() - time_start).total_seconds(),
    }

def _pca_w2(ref_coords, model_coords, n_atoms, K=None, prefix=""):
    # print("Calculating PCA-W2")
    if len(ref_coords.shape) == 3:
        ref_coords = ref_coords.reshape(ref_coords.shape[0], -1)
        model_coords = model_coords.reshape(model_coords.shape[0], -1)
    if K is not None:
        ref_coords = ref_coords[:,:K]
        model_coords = model_coords[:,:K]

    distmat = np.square(ref_coords[:,None] - model_coords[None]).sum(-1) 
    distmat = distmat ** 0.5 / n_atoms ** 0.5 * 10
    return {
        f'{prefix}pca_w2': _get_wasserstein(distmat),
    }

def _calc_cosine_sim(ref_pca, model_pca):
    # print("Calculating Cosine Sim")
    time_start = datetime.now()
    return {
        'cosine_sim': (ref_pca.components_[0] * model_pca.components_[0]).sum(),
        'cosine_sim_time_elapsed': (datetime.now() - time_start).total_seconds(),
    }

def _calc_weak_transient_contacts(ref_traj, model_traj, ref):
    # print("Calculating Weak Transient Contacts")
    time_start = datetime.now()
    ref_distmat = np.linalg.norm(ref_traj.xyz[:,None,:] - ref_traj.xyz[:,:,None], axis=-1)
    model_distmat = np.linalg.norm(model_traj.xyz[:,None,:] - model_traj.xyz[:,:,None], axis=-1)

    ref_contact_prob = (ref_distmat < 0.8).mean(0)
    model_contact_prob = (model_distmat < 0.8).mean(0)
    crystal_distmat = np.linalg.norm(ref.xyz[0,None,:] - ref.xyz[0,:,None], axis=-1)

    crystal_contact_mask = crystal_distmat < 0.8
    ref_transient_mask = (~crystal_contact_mask) & (ref_contact_prob > 0.1)
    model_transient_mask = (~crystal_contact_mask) & (model_contact_prob > 0.1)
    ref_weak_mask = crystal_contact_mask & (ref_contact_prob < 0.9)
    model_weak_mask = crystal_contact_mask & (model_contact_prob < 0.9)

    weak_contacts = (ref_weak_mask & model_weak_mask).sum() / (ref_weak_mask | model_weak_mask).sum()
    transient_contacts = (ref_transient_mask & model_transient_mask).sum() / (ref_transient_mask | model_transient_mask).sum()

    return {
        'weak_contacts': weak_contacts,
        'transient_contacts': transient_contacts,
        # 'ref_contact_prob': ref_contact_prob,
        # 'model_contact_prob': model_contact_prob,
        # 'crystal_distmat': crystal_distmat,
        'weak_transient_contacts_time_elapsed': (datetime.now() - time_start).total_seconds(),
    }

def _calc_clashes(traj, ref, threshold=3):
    # print("Calculating clashes")
    time_start = datetime.now()
    distmat = np.linalg.norm(traj.xyz[:,None,:,:] - traj.xyz[:,:,None,:], axis=-1)
    distmat = distmat * 10
    distmat[:, np.arange(traj.n_residues), np.arange(traj.n_residues)] = np.inf
    indices = np.triu_indices(distmat.shape[1], k=1)

    if threshold == "include_CACA_MD_dist":
        # some input equilibrium structures of ATLAS have neighbours
        # whose distance is around 3.2 A, instead of the normal 3.8.
        dist_CACA_ref = np.linalg.norm(ref.xyz[0,1:,:] - ref.xyz[0,:-1,:], axis=-1)
        dist_CACA_ref = dist_CACA_ref * 10
        small_CA_dist = dist_CACA_ref < 3.5
        default_threshold = 3.0
        CA_dist_threshold = np.ones((traj.n_frames, traj.n_residues-1)) * default_threshold
        CA_dist_threshold[:,small_CA_dist] = 2.8
        threshold = default_threshold * np.ones_like(distmat)
        threshold[:, np.arange(traj.n_residues-1), np.arange(1, traj.n_residues)] = CA_dist_threshold
        threshold[:, np.arange(1, traj.n_residues), np.arange(traj.n_residues-1)] = CA_dist_threshold
        threshold_flat = threshold.reshape(traj.n_frames, -1)
        threshold_nn = threshold[:, indices[0], indices[1]]
    else:
        threshold_flat = threshold * np.ones_like(distmat.reshape(traj.n_frames, -1))
        threshold_nn = threshold * np.ones_like(distmat[:, indices[0], indices[1]])

    conf_contains_clashes = np.any(distmat.reshape(traj.n_frames, -1) < threshold_flat, axis=1)

    distances = distmat[:, indices[0], indices[1]]

    return {
        'clashes_ratio_conf': conf_contains_clashes.mean(), # (num confs with clashes) / (num confs)
        'clashes_ratio_all': (distances < threshold_nn).mean(), # (num respairs with clashes) / (num respairs)
        'clashes_all': (distances < threshold_nn).sum(), # (num respairs with clashes)
        'clashes_time_elapsed': (datetime.now() - time_start).total_seconds(),
    }

def _calc_CACA_breaks(traj, ref_traj, threshold=4.19):
    # print("Calculating breaks")
    time_start = datetime.now()

    if threshold == "max_MD":
        threshold = np.max(np.linalg.norm(ref_traj.xyz[:,1:,:] - ref_traj.xyz[:,:-1,:], axis=-1)) * 10

    distances = np.linalg.norm(traj.xyz[:,1:,:] - traj.xyz[:,:-1,:], axis=-1)
    distances = distances * 10
    conf_contains_breaks = np.any(distances > threshold, axis=1)
    return {
        'breaks_CACA_ratio_conf': conf_contains_breaks.mean(), # (num confs with breaks) / (num confs)
        'breaks_CACA_ratio_all': (distances > threshold).mean(), # (num neighbours with breaks) / (num neighbours)
        'breaks_CACA_all': (distances > threshold).sum(), # (num neighbours with breaks)
        'breaks_CACA_time_elapsed': (datetime.now() - time_start).total_seconds(),
    }

def _calc_NC_breaks(traj, ref_traj, threshold=1.5):
    # print("Calculating breaks")
    time_start = datetime.now()

    if threshold == "max_MD":
        N_pos = ref_traj.xyz[:,ref_traj.top.select('name N'),:]
        C_pos = ref_traj.xyz[:,ref_traj.top.select('name C'),:]
        threshold = np.max(np.linalg.norm(N_pos[:,1:,:] - C_pos[:,:-1,:], axis=-1)) * 10

    N_pos = traj.xyz[:,traj.top.select('name N'),:]
    C_pos = traj.xyz[:,traj.top.select('name C'),:]
    distances = np.linalg.norm(N_pos[:,1:,:] - C_pos[:,:-1,:], axis=-1)
    distances = distances * 10
    conf_contains_breaks = np.any(distances > threshold, axis=1)
    return {
        'breaks_NC_ratio_conf': conf_contains_breaks.mean(), # (num confs with breaks) / (num confs)
        'breaks_NC_ratio_all': (distances > threshold).mean(), # (num neighbours with breaks) / (num neighbours)
        'breaks_NC_all': (distances > threshold).sum(), # (num neighbours with breaks)
        'breaks_NC_time_elapsed': (datetime.now() - time_start).total_seconds(),
    }
    

def _bootstrap(func, data, n_samples, n_iterations=100, alpha=0.05):
    sample_estimate = func(data, np.arange(n_samples))

    bootstrap_estimates = np.zeros((n_iterations, *sample_estimate.shape))
    for i in range(n_iterations):
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_estimates[i] = func(data, bootstrap_indices)

    low = 2 * sample_estimate - np.percentile(bootstrap_estimates, 100 * (1-alpha/2), axis=0)
    high = 2 * sample_estimate - np.percentile(bootstrap_estimates, 100 * (alpha/2), axis=0)

    return sample_estimate, low, high


def _load_model_traj(name, pdbdirs):
    model_traj_aa = None
    if isinstance(name, (tuple, list)):
        model_traj_path = name[0]
        if not os.path.exists(model_traj_path):
            warnings.warn(f"Expected model trajectory path {model_traj_path} does not exist. Skipping...")
            return None
        model_traj_aa = mdtraj.load(model_traj_path)

    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for pdbdir in pdbdirs:
                if os.path.exists(f'{pdbdir}/{name}/sampled_conformations.pdb'):
                    model_traj_aa = mdtraj.load(f'{pdbdir}/{name}/sampled_conformations.pdb')
                elif len(name) == 5 and os.path.exists(f'{pdbdir}/{name[:4]}_{name[4]}.pdb'):
                    model_traj_aa = mdtraj.load(f'{pdbdir}/{name[:4]}_{name[4]}.pdb')
                elif os.path.exists(f'{pdbdir}/{name}.pdb'):
                    model_traj_aa = mdtraj.load(f'{pdbdir}/{name}.pdb')
        if model_traj_aa is None:
            warnings.warn(f"No model samples found for {name} in {pdbdirs}. Skipping...")

    return model_traj_aa

def _load_ref_traj_full(name, reference_dirs, n_confs):
    # print(f"Loading {name}")
    ref_aa = ref_traj_aa = None
    if isinstance(name, (tuple, list)):
        if not all(os.path.exists(p) for p in name[1:]):
            warnings.warn(f"Expected reference trajectory paths {name[1:]} do not all exist. Skipping...")
            return None, None, None
        topfile = name[1]
        ref_traj_pahts = name[2:]
        ref_aa = mdtraj.load(topfile)
        ref_traj_aa = mdtraj.load(ref_traj_pahts[0], top=topfile)
        for path in ref_traj_pahts[1:]:
            ref_traj_aa += mdtraj.load(path, top=topfile)

    else:
        for reference_dir in reference_dirs:
            if not os.path.exists(f'{reference_dir}/{name}/{name}.pdb'):
                continue
            ref_aa = mdtraj.load(f'{reference_dir}/{name}/{name}.pdb')
            topfile = f'{reference_dir}/{name}/{name}.pdb'
            ref_traj_aa = mdtraj.load(f'{reference_dir}/{name}/{name}_R1.xtc', top=topfile) \
                + mdtraj.load(f'{reference_dir}/{name}/{name}_R2.xtc', top=topfile) \
                + mdtraj.load(f'{reference_dir}/{name}/{name}_R3.xtc', top=topfile)
    
    if ref_aa is None or ref_traj_aa is None:
        warnings.warn(f"No reference trajectory found for {name} in {reference_dirs}. Skipping...")
        return None, None, None
    
    RAND1 = np.random.randint(0, ref_traj_aa.n_frames, n_confs)
    RAND2 = np.random.randint(0, ref_traj_aa.n_frames, n_confs)
    RAND1K = np.random.randint(0, ref_traj_aa.n_frames, 1000)
    return ref_aa, ref_traj_aa, (RAND1, RAND2, RAND1K)

def _load_ref_traj_indices(paths, topfile, indices, n_frames_cumsum):
    frames = []
    for i, path in enumerate(paths):
        indices_i = indices[(indices >= n_frames_cumsum[i]) & (indices < n_frames_cumsum[i+1])]
        indices_i -= n_frames_cumsum[i]
        traj = mda.coordinates.XTC.XTCReader(path,refresh_offsets=True)
        traj = traj.trajectory
        frames_i = np.stack([traj[i].positions.copy() for i in indices_i], axis=0)
        frames.append(frames_i)

    frames = np.concatenate(frames, axis=0)
    return mdtraj.Trajectory(frames/10, mdtraj.load(topfile).topology)
def _load_ref_traj_subset(name, reference_dirs, n_confs):
    ref_aa = ref_traj_aa = None
    for reference_dir in reference_dirs:
        if not os.path.exists(f'{reference_dir}/{name}/{name}.pdb'):
            continue
        ref_aa = mdtraj.load(f'{reference_dir}/{name}/{name}.pdb')
        topfile = f'{reference_dir}/{name}/{name}.pdb'

        n_frames = [
            mda.coordinates.XTC.XTCReader(
                f'{reference_dir}/{name}/{name}_R{i+1}.xtc', refresh_offsets=True
            ).n_frames for i in range(3)
        ]
        RAND1 = np.random.randint(0, sum(n_frames), n_confs)
        RAND2 = np.random.randint(0, sum(n_frames), n_confs)
        RAND1K = np.random.randint(0, sum(n_frames), 1000)
        ref_traj_paths = [f'{reference_dir}/{name}/{name}_R{i+1}.xtc' for i in range(3)]
        n_frames_cumsum = (np.cumsum([0]+n_frames)).astype(int)
        ref_traj_1 = _load_ref_traj_indices(
            ref_traj_paths, topfile, RAND1, n_frames_cumsum
        )
        ref_traj_2 = _load_ref_traj_indices(
            ref_traj_paths, topfile, RAND2, n_frames_cumsum
        )
        ref_traj_1K = _load_ref_traj_indices(
            ref_traj_paths, topfile, RAND1K, n_frames_cumsum
        )
        ref_traj_aa = ref_traj_1 + ref_traj_2 + ref_traj_1K
    if ref_aa is None:
        print(f"No reference trajectory found for {name}")

    RAND1 = np.arange(0, n_confs)
    RAND2 = np.arange(n_confs, 2*n_confs)
    RAND1K = np.arange(2*n_confs, 2*n_confs+1000)
    return ref_aa, ref_traj_aa, (RAND1, RAND2, RAND1K)

def _slice_and_superpose(ref_aa, ref_traj_aa, model_traj_aa):
    time_start = datetime.now()

    ref_traj_aa.atom_slice([a.index for a in ref_traj_aa.top.atoms if a.element.symbol != 'H'], True)
    ref_aa.atom_slice([a.index for a in ref_aa.top.atoms if a.element.symbol != 'H'], True)
    model_traj_aa.atom_slice([a.index for a in model_traj_aa.top.atoms if a.element.symbol != 'H'], True)

    # only backbone atoms, since our model only predicts backbone atoms
    model_traj_aa.atom_slice([a.index for a in model_traj_aa.top.atoms if a.name in ['CA', 'C', 'N', 'O', 'OXT']], True)

    refmask, modelmask = _align_tops(ref_traj_aa.top, model_traj_aa.top, True)
    ref_traj_aa.atom_slice(refmask, True)
    ref_aa.atom_slice(refmask, True)
    model_traj_aa.atom_slice(modelmask, True)

    ref_traj_aa.superpose(ref_aa, parallel=False)
    model_traj_aa.superpose(ref_aa, parallel=False)

    ca_mask = [a.index for a in ref_traj_aa.top.atoms if a.name == 'CA']
    ca_mask_model = [a.index for a in model_traj_aa.top.atoms if a.name == 'CA']
    ref_traj = ref_traj_aa.atom_slice(ca_mask, False)
    ref = ref_aa.atom_slice(ca_mask, False)
    model_traj = model_traj_aa.atom_slice(ca_mask_model, False)
    
    ref_traj.superpose(ref, parallel=False)
    model_traj.superpose(ref, parallel=False)

    slice_superpose_time_elapsed = (datetime.now() - time_start).total_seconds()

    return (
        (ref_aa, ref), 
        (ref_traj_aa, ref_traj),
        (model_traj_aa, model_traj),
        slice_superpose_time_elapsed
    )

def _calc_metrics(
        name, metrics_subset, load_subset,
        ref_aa, ref,
        ref_traj_aa, ref_traj,
        model_traj_aa, model_traj,
        RAND1, RAND2, RAND1K,
    ):
    n_atoms = ref_traj.n_atoms

    # ref_traj_aa_RAND1 = ref_traj_aa[RAND1] # not used
    # ref_traj_aa_RAND2 = ref_traj_aa[RAND2] # not used
    ref_traj_aa_RAND1K = ref_traj_aa[RAND1K]
    ref_traj_RAND1 = ref_traj[RAND1]
    ref_traj_RAND2 = ref_traj[RAND2]
    ref_traj_RAND1K = ref_traj[RAND1K]
    if load_subset:
        ref_traj = ref_traj[RAND1K]
        ref_traj_aa = ref_traj_aa[RAND1K]

    out = {}

    out["length"] = ref_traj.n_residues
    out["num_conformations"] = model_traj.n_frames

    if metrics_subset is None or "RMSD" in metrics_subset:
        out.update(_calc_rmsd(ref_traj_RAND1, ref_traj_RAND2, model_traj))

    if metrics_subset is None or "RMSF" in metrics_subset:
        time_start = datetime.now()

        # CA_rmsf with bootstrap for plotting.
        ref_traj_cloned = mdtraj.Trajectory(ref_traj.xyz.copy(), ref_traj.top)
        model_traj_cloned = mdtraj.Trajectory(model_traj.xyz.copy(), model_traj.top)
        ref_cloned = mdtraj.Trajectory(ref.xyz.copy(), ref.top)
        CA_ref_rmsf = _calc_rmsf(ref_traj_cloned, ref_cloned, bootstrap=True)
        CA_model_rmsf = _calc_rmsf(model_traj_cloned, ref_cloned, bootstrap=True)
        out.update({f"CA_ref_{k}": v for k, v in CA_ref_rmsf.items()})
        out.update({f"CA_model_{k}": v for k, v in CA_model_rmsf.items()})

        # rmsf on CA only without bootstrap for analysis
        ref_traj_cloned = mdtraj.Trajectory(ref_traj.xyz.copy(), ref_traj.top)
        model_traj_cloned = mdtraj.Trajectory(model_traj.xyz.copy(), model_traj.top)
        ref_cloned = mdtraj.Trajectory(ref.xyz.copy(), ref.top)
        out["ref_rmsf"] = _calc_rmsf(ref_traj_cloned, ref_cloned, bootstrap=False)
        out["model_rmsf"] = _calc_rmsf(model_traj_cloned, ref_cloned, bootstrap=False)

        out['rmsf_time_elapsed'] = (datetime.now() - time_start).total_seconds()

    if metrics_subset is None or "BB-RMSF" in metrics_subset:
        time_start = datetime.now()

        # rmsf on backbone and full reference trajectory for analysis
        ref_traj_cloned = mdtraj.Trajectory(ref_traj_aa.xyz.copy(), ref_traj_aa.top)
        model_traj_cloned = mdtraj.Trajectory(model_traj_aa.xyz.copy(), model_traj_aa.top)
        ref_cloned = mdtraj.Trajectory(ref_aa.xyz.copy(), ref_aa.top)
        out["BB_ref_rmsf"] = _calc_rmsf(ref_traj_cloned, ref_cloned, bootstrap=False)
        out["BB_model_rmsf"] = _calc_rmsf(model_traj_cloned, ref_cloned, bootstrap=False)

        out['bb_rmsf_time_elapsed'] = (datetime.now() - time_start).total_seconds()

    if metrics_subset is None or "DCCM" in metrics_subset:
        out.update(_calc_dccm(ref_traj, model_traj))

    if metrics_subset is None or "RMWD" in metrics_subset:
        out.update(_calc_rmwd(ref_traj_RAND1K, model_traj))
        rmwd_bb = _calc_rmwd(ref_traj_aa_RAND1K, model_traj_aa)
        out.update({f"BB_{k}": v for k, v in rmwd_bb.items()})

    ref_pca = model_pca = None
    if metrics_subset is None or "PCA-W2" in metrics_subset:
        time_start = datetime.now()
        ref_pca, ref_coords = _get_pca(ref_traj.xyz)
        model_pca, _ = _get_pca(model_traj.xyz)
        model_coords_ref_pca = ref_pca.transform(model_traj.xyz.reshape(model_traj.n_frames, -1))
        joint_pca, _ = _get_pca(np.concatenate([ref_traj_RAND1.xyz, model_traj.xyz]))
        ref_coords_joint_pca = joint_pca.transform(ref_traj.xyz.reshape(ref_traj.n_frames, -1))
        model_coords_joint_pca = joint_pca.transform(model_traj.xyz.reshape(model_traj.n_frames, -1))
        
        out['ref_variance'] = ref_pca.explained_variance_ / n_atoms * 100
        out['model_variance'] = model_pca.explained_variance_ / n_atoms * 100
        out['joint_variance'] = joint_pca.explained_variance_ / n_atoms * 100

        out['md_pca_ref_coords'] = ref_coords
        out['md_pca_model_coords'] = model_coords_ref_pca
        out['joint_pca_ref_coords'] = ref_coords_joint_pca
        out['joint_pca_model_coords'] = model_coords_joint_pca

        RAND1 = np.random.randint(0, ref_traj.n_frames, model_traj.n_frames)
        out.update(_pca_w2(ref_coords[RAND1], model_coords_ref_pca, n_atoms, K=2, prefix="md_"))
        out.update(_pca_w2(ref_coords_joint_pca[RAND1], model_coords_joint_pca, n_atoms, K=2, prefix="joint_"))
        out['pca_w2_time_elapsed'] = (datetime.now() - time_start).total_seconds()
        
    if metrics_subset is None or "Cosine Sim" in metrics_subset:
        if ref_pca is None:
            ref_pca, _ = _get_pca(ref_traj.xyz)
            model_pca, _ = _get_pca(model_traj.xyz)
        out.update(_calc_cosine_sim(ref_pca, model_pca))

    if metrics_subset is None or "Weak Transient Contacts" in metrics_subset:
        out.update(_calc_weak_transient_contacts(ref_traj_RAND1, model_traj, ref))

    if metrics_subset is None or "Breaks and Clashes" in metrics_subset:
        out.update(_calc_clashes(model_traj, ref, threshold="include_CACA_MD_dist"))
        out.update(_calc_CACA_breaks(model_traj, ref_traj, threshold="max_MD"))
        out.update(_calc_NC_breaks(model_traj_aa, ref_traj_aa, threshold="max_MD"))
    
    return out
def _calc_metrics_bootstrap(
        name, reference_dirs, pdbdirs, num_conformations=None, num_bootstrap_samples=1,
        seed=None, load_subset=False, metrics_subset=None,
        load_ref_fn=_load_ref_traj_full, load_model_fn=_load_model_traj,
    ):
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed((os.getpid() * hash(name) * int(time.time())) % 2**32)

    time_start = datetime.now()
    # load reference and model trajectories
    model_traj_aa = load_model_fn(name, pdbdirs)
    if model_traj_aa is None:
        return name, None
    ref_aa, ref_traj_aa, rand_indices = load_ref_fn(
        name, reference_dirs, 
        model_traj_aa.n_frames if num_conformations is None else num_conformations
    )
    if ref_aa is None or ref_traj_aa is None:
        return name, None
    loading_traj_time_elapsed = (datetime.now() - time_start).total_seconds()
    RAND1, RAND2, RAND1K = rand_indices

    ref, ref_traj, model_traj, slice_superpose_time_elapsed = _slice_and_superpose(
        ref_aa, ref_traj_aa, model_traj_aa
    )
    ref_aa, ref = ref
    ref_traj_aa, ref_traj = ref_traj
    model_traj_aa, model_traj = model_traj

    
    if num_bootstrap_samples == 1:
        if num_conformations is not None and num_conformations < model_traj_aa.n_frames:
            model_indices = np.random.choice(
                model_traj_aa.n_frames, num_conformations, replace=False
            )
            model_traj_aa = model_traj_aa[model_indices]
            model_traj = model_traj[model_indices]
            
            RAND1 = np.random.randint(0, ref_traj_aa.n_frames, model_traj_aa.n_frames)
            RAND2 = np.random.randint(0, ref_traj_aa.n_frames, model_traj_aa.n_frames)
            RAND1K = np.random.randint(0, ref_traj_aa.n_frames, 1000)

        out = _calc_metrics(
            name, metrics_subset, load_subset,
            ref_aa, ref,
            ref_traj_aa, ref_traj,
            model_traj_aa, model_traj,
            RAND1, RAND2, RAND1K,
        )
        out["loading_traj_time_elapsed"] = loading_traj_time_elapsed
        out["slice_superpose_time_elapsed"] = slice_superpose_time_elapsed
        return name, out
    

    else:
    
        out = {}
        for i in range(num_bootstrap_samples):
            if num_conformations is not None and num_conformations < model_traj_aa.n_frames:
                model_indices = np.random.choice(
                    model_traj_aa.n_frames, num_conformations, replace=False
                )
                model_traj_aa_i = model_traj_aa[model_indices]
                model_traj_i = model_traj[model_indices]
            else:
                model_traj_aa_i = model_traj_aa
                model_traj_i = model_traj
                
            RAND1 = np.random.randint(0, ref_traj_aa.n_frames, model_traj_aa_i.n_frames)
            RAND2 = np.random.randint(0, ref_traj_aa.n_frames, model_traj_aa_i.n_frames)
            RAND1K = np.random.randint(0, ref_traj_aa.n_frames, 1000)

            out_i = _calc_metrics(
                name, metrics_subset, load_subset,
                ref_aa, ref,
                ref_traj_aa, ref_traj,
                model_traj_aa_i, model_traj_i,
                RAND1, RAND2, RAND1K,
            )
            for key, val in out_i.items():
                if key not in out:
                    out[key] = []
                out[key].append(val)

        for key, val in out.items():
            out[key] = np.array(val)
        out["loading_traj_time_elapsed"] = loading_traj_time_elapsed*np.ones(num_bootstrap_samples)
        out["slice_superpose_time_elapsed"] = slice_superpose_time_elapsed*np.ones(num_bootstrap_samples)


        return name, out

def _analyze_data(data, metrics_subset=None):
    # print("Analyzing data")
    concatenated_lists = {}
    if metrics_subset is None or "RMSF" in metrics_subset:
        concatenated_lists["rmsf"] = {
            "ref": np.concatenate([data[name]['ref_rmsf'] for name in data]),
            "model": np.concatenate([data[name]['model_rmsf'] for name in data]),
        }
    if metrics_subset is None or "BB-RMSF" in metrics_subset:
        concatenated_lists["BB_rmsf"] = {
            "ref": np.concatenate([data[name]['BB_ref_rmsf'] for name in data]),
            "model": np.concatenate([data[name]['BB_model_rmsf'] for name in data]),
        }
    if metrics_subset is None or "DCCM" in metrics_subset:
        concatenated_lists["dccm"] = {
            "ref": np.concatenate([data[name]['ref_dccm'].flatten() for name in data]),
            "model": np.concatenate([data[name]['model_dccm'].flatten() for name in data]),
        }
    if metrics_subset is None or "Breaks and Clashes" in metrics_subset:
        total_number_of_clashes = np.sum([data[name]['clashes_all'] for name in data])
        total_number_of_confs_with_clashes = np.sum([data[name]['clashes_ratio_conf']*data[name]['num_conformations'] for name in data])
        total_number_of_nn_pairs = sum([int(data[name]['length']*(data[name]['length']-1)/2) for name in data]) 
        total_number_of_breaks_CACA = np.sum([data[name]['breaks_CACA_all'] for name in data])
        total_number_of_confs_with_breaks_CACA = np.sum([data[name]['breaks_CACA_ratio_conf']*data[name]['num_conformations'] for name in data])
        total_number_of_breaks_NC = np.sum([data[name]['breaks_NC_all'] for name in data])
        total_number_of_confs_with_breaks_NC = np.sum([data[name]['breaks_NC_ratio_conf']*data[name]['num_conformations'] for name in data])
        total_number_of_neighbours = sum([data[name]['length']-1 for name in data])
        total_number_of_confs = sum([data[name]['num_conformations'] for name in data])
        
    df = []

    for name, out in data.items():
        item = {
            "name": name,
            "loading_traj_time_elapsed": out["loading_traj_time_elapsed"],
            "slice_superpose_time_elapsed": out["slice_superpose_time_elapsed"]
        }
        if metrics_subset is None or "RMSD" in metrics_subset:
            item.update({
                "md_pairwise_rmsd": out['ref_mean_pairwise_rmsd'],
                "model_pairwise_rmsd": out['model_mean_pairwise_rmsd'],
                "rmsd_time_elapsed": out['rmsd_time_elapsed'],
            })

        if metrics_subset is None or "RMSF" in metrics_subset:
            item["per_target_rmsf_corr"] = _correlations(
                out['model_rmsf'],
                out['ref_rmsf'], 
                correlation_type="pearson"
            )
            item["per_target_rmsf_mae"] = np.abs(out['model_rmsf'] - out['ref_rmsf']).mean()
            item["per_target_rmsf_rmse"] = np.sqrt(((out['model_rmsf'] - out['ref_rmsf']) ** 2).mean())
            item["rmsf_time_elapsed"] = out['rmsf_time_elapsed']

        if metrics_subset is None or "BB-RMSF" in metrics_subset:
            item["per_target_rmsf_corr_bb"] = _correlations(
                out['BB_model_rmsf'],
                out['BB_ref_rmsf'], 
                correlation_type="pearson"
            )
            item["per_target_rmsf_mae_bb"] = np.abs(out['BB_model_rmsf'] - out['BB_ref_rmsf']).mean()
            item["per_target_rmsf_rmse_bb"] = np.sqrt(((out['BB_model_rmsf'] - out['BB_ref_rmsf']) ** 2).mean())
            item["rmsf_time_elapsed"] = out['rmsf_time_elapsed']

        if metrics_subset is None or "DCCM" in metrics_subset:
            item["per_target_dccm_corr"] = _correlations(
                out['model_dccm'].flatten(),
                out['ref_dccm'].flatten(),
                correlation_type="pearson"
            )
            item["per_target_dccm_mae"] = np.abs(out['model_dccm'].flatten() - out['ref_dccm'].flatten()).mean()
            item["per_target_dccm_rmse"] = np.sqrt(((out['model_dccm'].flatten() - out['ref_dccm'].flatten()) ** 2).mean())
            item["dccm_time_elapsed"] = out['dccm_time_elapsed']

        if metrics_subset is None or "RMWD" in metrics_subset:
            item.update({
                'rmwd': out['rmwd'],
                'rmwd_trans': out['rmwd_trans'],
                'rmwd_var': out['rmwd_var'],
                'BB_rmwd': out['BB_rmwd'],
                'BB_rmwd_trans': out['BB_rmwd_trans'],
                'BB_rmwd_var': out['BB_rmwd_var'],
                'rmwd_time_elapsed': out['rmwd_time_elapsed']+out['BB_rmwd_time_elapsed'],
            })

        if metrics_subset is None or "PCA-W2" in metrics_subset:
            item.update({
                'md_pca_w2': out['md_pca_w2'],
                'joint_pca_w2': out['joint_pca_w2'],
                'pca_w2_time_elapsed': out['pca_w2_time_elapsed'],
            })

        if metrics_subset is None or "Cosine Sim" in metrics_subset:
            item['cosine_sim'] = out['cosine_sim']
            item['cosine_sim_time_elapsed'] = out['cosine_sim_time_elapsed']

        if metrics_subset is None or "Weak Transient Contacts" in metrics_subset:
            item.update({
                'weak_contacts': out['weak_contacts'],
                'transient_contacts': out['transient_contacts'],
                'weak_transient_contacts_time_elapsed': out['weak_transient_contacts_time_elapsed'],
            })

        if metrics_subset is None or "Breaks and Clashes" in metrics_subset:
            item.update({
                'clashes_ratio_conf': out['clashes_ratio_conf'],
                'clashes_ratio_all': out['clashes_ratio_all'],
                'clashes_all': out['clashes_all'],
                'breaks_CACA_ratio_conf': out['breaks_CACA_ratio_conf'],
                'breaks_CACA_ratio_all': out['breaks_CACA_ratio_all'],
                'breaks_CACA_all': out['breaks_CACA_all'],
                'breaks_NC_ratio_conf': out['breaks_NC_ratio_conf'],
                'breaks_NC_ratio_all': out['breaks_NC_ratio_all'],
                'breaks_NC_all': out['breaks_NC_all'],
                'breaks_and_clashes_time_elapsed': out['clashes_time_elapsed'] + out['breaks_CACA_time_elapsed'] + out['breaks_NC_time_elapsed']
            })

        df.append(item)

    df = pd.DataFrame(df).set_index('name')

    metrics_dict = {}
    if metrics_subset is None or "RMSD" in metrics_subset:
        metrics_dict["MD pairwise RMSD"] = df.md_pairwise_rmsd.median()
        metrics_dict["Pairwise RMSD"] = df.model_pairwise_rmsd.median()
        metrics_dict["MD pairwise RMSD mean"] = df.md_pairwise_rmsd.mean()
        metrics_dict["Pairwise RMSD mean"] = df.model_pairwise_rmsd.mean()
        metrics_dict["Pairwise RMSD r"] = _correlations(df.md_pairwise_rmsd, df.model_pairwise_rmsd, "pearson")
        metrics_dict["Pairwise RMSD MAE"] = np.abs(df.md_pairwise_rmsd - df.model_pairwise_rmsd).mean()
        metrics_dict["Pairwise RMSD RMSE"] = np.sqrt(((df.md_pairwise_rmsd - df.model_pairwise_rmsd) ** 2).mean())
        metrics_dict["Pairwise RMSD MAE max"] = np.abs(df.md_pairwise_rmsd - df.model_pairwise_rmsd).max()
        metrics_dict["Pairwise RMSD RMSE max"] = np.sqrt(((df.md_pairwise_rmsd - df.model_pairwise_rmsd) ** 2).max())

    if metrics_subset is None or "RMSF" in metrics_subset:
        metrics_dict["MD RMSF"] = np.median(concatenated_lists["rmsf"]["ref"])
        metrics_dict["RMSF"] = np.median(concatenated_lists["rmsf"]["model"])
        metrics_dict["MD RMSF mean"] = np.mean(concatenated_lists["rmsf"]["ref"])
        metrics_dict["RMSF mean"] = np.mean(concatenated_lists["rmsf"]["model"])
        metrics_dict["Global RMSF r"] = _correlations(concatenated_lists["rmsf"]["ref"], concatenated_lists["rmsf"]["model"], "pearson")
        metrics_dict["Global RMSF MAE"] = np.abs(concatenated_lists["rmsf"]["ref"] - concatenated_lists["rmsf"]["model"]).mean()
        metrics_dict["Global RMSF RMSE"] = np.sqrt(((concatenated_lists["rmsf"]["ref"] - concatenated_lists["rmsf"]["model"]) ** 2).mean())
        metrics_dict["Per target RMSF r"] = df.per_target_rmsf_corr.median()
        metrics_dict["Per target RMSF r mean"] = df.per_target_rmsf_corr.mean()
        metrics_dict["Per target RMSF r min"] = df.per_target_rmsf_corr.min()
        metrics_dict["Per target RMSF MAE"] = df.per_target_rmsf_mae.median()
        metrics_dict["Per target RMSF MAE mean"] = df.per_target_rmsf_mae.mean()
        metrics_dict["Per target RMSF MAE max"] = df.per_target_rmsf_mae.max()
        metrics_dict["Per target RMSF RMSE"] = df.per_target_rmsf_rmse.median()
        metrics_dict["Per target RMSF RMSE mean"] = df.per_target_rmsf_rmse.mean()
        metrics_dict["Per target RMSF RMSE max"] = df.per_target_rmsf_rmse.max()

    if metrics_subset is None or "BB-RMSF" in metrics_subset:
        metrics_dict["MD BB-RMSF"] = np.median(concatenated_lists["BB_rmsf"]["ref"])
        metrics_dict["BB-RMSF"] = np.median(concatenated_lists["BB_rmsf"]["model"])
        metrics_dict["MD BB-RMSF mean"] = np.mean(concatenated_lists["BB_rmsf"]["ref"])
        metrics_dict["BB-RMSF mean"] = np.mean(concatenated_lists["BB_rmsf"]["model"])
        metrics_dict["Global BB-RMSF r"] = _correlations(concatenated_lists["BB_rmsf"]["ref"], concatenated_lists["BB_rmsf"]["model"], "pearson")
        metrics_dict["Global BB-RMSF MAE"] = np.abs(concatenated_lists["BB_rmsf"]["ref"] - concatenated_lists["BB_rmsf"]["model"]).mean()
        metrics_dict["Global BB-RMSF RMSE"] = np.sqrt(((concatenated_lists["BB_rmsf"]["ref"] - concatenated_lists["BB_rmsf"]["model"]) ** 2).mean())
        metrics_dict["Per target BB-RMSF r"] = df.per_target_rmsf_corr_bb.median()
        metrics_dict["Per target BB-RMSF r mean"] = df.per_target_rmsf_corr_bb.mean()
        metrics_dict["Per target BB-RMSF r min"] = df.per_target_rmsf_corr_bb.min()
        metrics_dict["Per target BB-RMSF MAE"] = df.per_target_rmsf_mae_bb.median()
        metrics_dict["Per target BB-RMSF MAE mean"] = df.per_target_rmsf_mae_bb.mean()
        metrics_dict["Per target BB-RMSF MAE max"] = df.per_target_rmsf_mae_bb.max()
        metrics_dict["Per target BB-RMSF RMSE"] = df.per_target_rmsf_rmse_bb.median()
        metrics_dict["Per target BB-RMSF RMSE mean"] = df.per_target_rmsf_rmse_bb.mean()
        metrics_dict["Per target BB-RMSF RMSE max"] = df.per_target_rmsf_rmse_bb.max()

    if metrics_subset is None or "DCCM" in metrics_subset:
        metrics_dict["MD DCCM"] = np.median(concatenated_lists["dccm"]["ref"])
        metrics_dict["DCCM"] = np.median(concatenated_lists["dccm"]["model"])
        metrics_dict["MD DCCM mean"] = np.mean(concatenated_lists["dccm"]["ref"])
        metrics_dict["DCCM mean"] = np.mean(concatenated_lists["dccm"]["model"])
        metrics_dict["Global DCCM r"] = _correlations(concatenated_lists["dccm"]["ref"], concatenated_lists["dccm"]["model"], "pearson")
        metrics_dict["Global DCCM MAE"] = np.abs(concatenated_lists["dccm"]["ref"] - concatenated_lists["dccm"]["model"]).mean()
        metrics_dict["Global DCCM RMSE"] = np.sqrt(((concatenated_lists["dccm"]["ref"] - concatenated_lists["dccm"]["model"]) ** 2).mean())
        metrics_dict["Per target DCCM r"] = df.per_target_dccm_corr.median()
        metrics_dict["Per target DCCM r mean"] = df.per_target_dccm_corr.mean()
        metrics_dict["Per target DCCM r min"] = df.per_target_dccm_corr.min()
        metrics_dict["Per target DCCM MAE"] = df.per_target_dccm_mae.median()
        metrics_dict["Per target DCCM MAE mean"] = df.per_target_dccm_mae.mean()
        metrics_dict["Per target DCCM MAE max"] = df.per_target_dccm_mae.max()
        metrics_dict["Per target DCCM RMSE"] = df.per_target_dccm_rmse.median()
        metrics_dict["Per target DCCM RMSE mean"] = df.per_target_dccm_rmse.mean()
        metrics_dict["Per target DCCM RMSE max"] = df.per_target_dccm_rmse.max()

    if metrics_subset is None or "RMWD" in metrics_subset:
        metrics_dict["RMWD"] = df.rmwd.median()
        metrics_dict["RMWD translation"] = df.rmwd_trans.median()
        metrics_dict["RMWD variance"] = df.rmwd_var.median()

        metrics_dict["BB RMWD"] = df.BB_rmwd.median()
        metrics_dict["BB RMWD translation"] = df.BB_rmwd_trans.median()
        metrics_dict["BB RMWD variance"] = df.BB_rmwd_var.median()

    if metrics_subset is None or "PCA-W2" in metrics_subset:
        metrics_dict["MD PCA W2"] = df.md_pca_w2.median()
        metrics_dict["Joint PCA W2"] = df.joint_pca_w2.median()
        
    if metrics_subset is None or "Cosine Sim" in metrics_subset:
        metrics_dict["% PC sim > 0.5"] = (df.cosine_sim.abs() > 0.5).mean() * 100

    if metrics_subset is None or "Weak Transient Contacts" in metrics_subset:
        metrics_dict["Weak contacts J"] = df.weak_contacts.median()
        metrics_dict["Weak contacts nans"] = df.weak_contacts.isna().mean()
        metrics_dict["Transient contacts J"] = df.transient_contacts.median()
        metrics_dict["Transient contacts nans"] = df.transient_contacts.isna().mean()

    if metrics_subset is None or "Breaks and Clashes" in metrics_subset:
        metrics_dict["Conf \\w clash ratio"] = df.clashes_ratio_conf.median() * 100
        metrics_dict["Conf \\w clash ratio mean"] = df.clashes_ratio_conf.mean() * 100
        metrics_dict["Conf \\w clash ratio max"] = df.clashes_ratio_conf.max() * 100
        metrics_dict["Per target clash ratio"] = df.clashes_ratio_all.median() * 100
        metrics_dict["Per target clash ratio mean"] = df.clashes_ratio_all.mean() * 100
        metrics_dict["Per target clash ratio max"] = df.clashes_ratio_all.max() * 100
        metrics_dict["Clash ratio"] = total_number_of_clashes / (total_number_of_nn_pairs * total_number_of_confs) * 100
        metrics_dict["Conf \\w clash ratio total"] = total_number_of_confs_with_clashes / total_number_of_confs * 100

        metrics_dict["Conf \\w CACA breaks ratio"] = df.breaks_CACA_ratio_conf.median() * 100
        metrics_dict["Conf \\w CACA breaks ratio mean"] = df.breaks_CACA_ratio_conf.mean() * 100
        metrics_dict["Conf \\w CACA breaks ratio max"] = df.breaks_CACA_ratio_conf.max() * 100
        metrics_dict["Per target CACA breaks ratio"] = df.breaks_CACA_ratio_all.median() * 100
        metrics_dict["Per target CACA breaks ratio mean"] = df.breaks_CACA_ratio_all.mean() * 100
        metrics_dict["Per target CACA breaks ratio max"] = df.breaks_CACA_ratio_all.max() * 100
        metrics_dict["CACA breaks ratio"] = total_number_of_breaks_CACA / (total_number_of_neighbours * total_number_of_confs) * 100
        metrics_dict["Conf \\w CACA breaks ratio total"] = total_number_of_confs_with_breaks_CACA / total_number_of_confs * 100

        metrics_dict["Conf \\w NC breaks ratio"] = df.breaks_NC_ratio_conf.median() * 100
        metrics_dict["Conf \\w NC breaks ratio mean"] = df.breaks_NC_ratio_conf.mean() * 100
        metrics_dict["Conf \\w NC breaks ratio max"] = df.breaks_NC_ratio_conf.max() * 100
        metrics_dict["Per target NC breaks ratio"] = df.breaks_NC_ratio_all.median() * 100
        metrics_dict["Per target NC breaks ratio mean"] = df.breaks_NC_ratio_all.mean() * 100
        metrics_dict["Per target NC breaks ratio max"] = df.breaks_NC_ratio_all.max() * 100
        metrics_dict["NC breaks ratio"] = total_number_of_breaks_NC / (total_number_of_neighbours * total_number_of_confs) * 100
        metrics_dict["Conf \\w NC breaks ratio total"] = total_number_of_confs_with_breaks_NC / total_number_of_confs * 100


    #for col in df.columns:
    #    if col.endswith("_time_elapsed"):
    #        print(col, df[col].sum())
    #        # metrics_dict[col] = df[col].sum()

    return metrics_dict


def calc_metrics(
        pdb_names, reference_dirs, pdbdirs, seed, num_workers, 
        num_conformations=None, num_bootstrap_samples=1,
        load_subset=False, metrics_subset=None, 
        load_ref_fn=_load_ref_traj_full, load_model_fn=_load_model_traj,
        use_tqdm=False
    ):
    if type(reference_dirs) == str:
        reference_dirs = [reference_dirs]
    if type(pdbdirs) == str:
        pdbdirs = [pdbdirs]
    print("Calc metrics:", metrics_subset)

    if num_workers > 1:
        p = Pool(num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map

    _calc_metrics_partial = functools.partial(
        _calc_metrics_bootstrap, 
        reference_dirs=reference_dirs, 
        pdbdirs=pdbdirs,
        num_conformations=num_conformations,
        num_bootstrap_samples=num_bootstrap_samples,
        seed=seed,
        load_subset=load_subset,
        load_ref_fn=load_ref_fn,
        load_model_fn=load_model_fn,
        metrics_subset=metrics_subset
    )
    if use_tqdm:
        out = dict(tqdm(__map__(_calc_metrics_partial, pdb_names), total=len(pdb_names), leave=False))
    else:
        out = dict(__map__(_calc_metrics_partial, pdb_names))
    if num_workers > 1:
        p.__exit__(None, None, None)

    out = {k: v for k, v in out.items() if v is not None}
    if len(out) == 0:
        raise ValueError("No valid trajectories were loaded.")
    
    if num_bootstrap_samples == 1:
        metrics_dict = _analyze_data(out, metrics_subset)
        return out, metrics_dict, None

    else:
        metrics_dict_bootstrap = {}
        for i in range(num_bootstrap_samples):
            out_i = {
                name: {k: v[i] for k, v in out[name].items()} for name in out
            }
            metrics_dict_i = _analyze_data(out_i, metrics_subset)
            for key, val in metrics_dict_i.items():
                if key not in metrics_dict_bootstrap:
                    metrics_dict_bootstrap[key] = []
                metrics_dict_bootstrap[key].append(val)

        metrics_dict_mean = {}
        metrics_dict_sem = {}
        metrics_dict_low = {}
        metrics_dict_high = {}
        for key, val in metrics_dict_bootstrap.items():
            val = np.array(val)
            ci = bootstrap((val,), np.mean, confidence_level=.95, vectorized=True)
            metrics_dict_mean[key] = val.mean()
            metrics_dict_sem[key] = ci.standard_error
            metrics_dict_low[key] = ci.confidence_interval.low
            metrics_dict_high[key] = ci.confidence_interval.high
            
        metrics_dict = {
            "mean": metrics_dict_mean,
            "sem": metrics_dict_sem,
            "low": metrics_dict_low,
            "high": metrics_dict_high,
            "bootstrap_samples": metrics_dict_bootstrap,
        }

        return out, metrics_dict, None



