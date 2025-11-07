import argparse
import os
import pickle
import pandas as pd
import numpy as np
import functools
from multiprocessing import Pool
from tqdm import tqdm
from scipy.stats import bootstrap
from typing import List, Tuple, Dict, Union, Optional

from bbflow.analysis.analyse_bbflow import (
    _load_ref_traj_full,
    _load_ref_traj_subset,
    _load_model_traj,
    _calc_metrics_bootstrap,
    _analyze_data,
)

default_metrics = [
    "RMSD",
    "RMSF",
    "DCCM",
    "RMWD",
    "PCA-W2",
    "Cosine Sim",
    "Weak Transient Contacts",
]

def calc_metrics(
    names: Union[List[str]]=None,
    md_dirs: Optional[List[str]]=None,
    ensemble_dirs: Optional[List[str]]=None,
    paths: Optional[List[Tuple]]=None,
    num_workers: int=8,
    metrics_subset: Optional[List[str]]=None,
    use_tqdm: bool=True,
    analysis_data_path: str="",
    analysis_metrics_path: str="",
    print_metrics: bool=True,
):
    """
    Calculate metrics for protein ensembles against reference MD trajectories.

    Either provide:
        1. names, md_dirs and ensemble_dirs
        or
        2. paths: list of tuples, each containing 
                (model_ensemble_pdb, reference_pdb, reference_traj1, ..., reference_trajN)

    Args:
        names:
            List of names of the ensembles to analyze. For each name, there must
            be a corresponding pdb file or directory in each of the ensemble_dirs,
            i.e. `ensemble_dirs[i]/name.pdb` or `ensemble_dirs[i]/name/sampled_conformations.pdb`,
            as well as a corresponding reference pdb and trajs in each of the md_dirs,
            i.e. `md_dirs[i]/name.pdb` and `md_dirs[i]/name/name_traj_R{j}.xtc` for j=1,2,3.
        md_dirs:
            List of directories containing reference MD data. Each directory must contain
            reference pdbs and 3 trajectories.
        ensemble_dirs:
            List of directories containing model ensembles. Each directory must contain
            pdb files or directories for each ensemble.
        paths:
            List of tuples, each containing 
            (model_ensemble.pdb, reference.pdb, reference_traj1.xtc, ..., reference_trajN.xtc)
        num_workers:
            Number of parallel workers to use.
        metrics_subset:
            Subset of metrics to calculate. If None, all metrics are calculated.
        use_tqdm:
            Whether to use tqdm progress bar.
        analysis_data_path:
            Path to save raw analysis data (pickle file). If empty, no file is saved.
        analysis_metrics_path:
            Path to save metrics summary (CSV file). If empty, no file is saved.
        print_metrics:
            Whether to print metrics to console.
    """
    if paths is not None:
        if names is not None or md_dirs is not None or ensemble_dirs is not None:
            raise ValueError("Either provide paths OR names, md_dirs and ensemble_dirs.")
        names = paths

    if metrics_subset is None:
        metrics_subset = default_metrics
    
    if num_workers > 1:
        p = Pool(num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map

    _calc_metrics_partial = functools.partial(
        _calc_metrics_bootstrap, 
        reference_dirs=md_dirs, 
        pdbdirs=ensemble_dirs,
        metrics_subset=metrics_subset
    )
    if use_tqdm:
        out = dict(tqdm(__map__(_calc_metrics_partial, names), total=len(names), leave=False))
    else:
        out = dict(__map__(_calc_metrics_partial, names))
    if num_workers > 1:
        p.__exit__(None, None, None)

    out = {k: v for k, v in out.items() if v is not None}
    if len(out) == 0:
        raise ValueError("No valid trajectories were loaded.")

    metrics_dict = _analyze_data(out, metrics_subset)

    if analysis_data_path != "":
        if paths is not None:
            # replace keys with range(len(paths)), otherwise keys are unreadable tuples
            out = {f"path_{i}": out[k] for i, k in enumerate(out.keys())}
        with open(analysis_data_path, 'wb') as f:
            pickle.dump(out, f)

    if analysis_metrics_path != "":
        df = pd.DataFrame({"": metrics_dict}).round(4)
        df = df.reindex([k for k in metrics_dict.keys()])
        df.to_csv(analysis_metrics_path)

    if print_metrics:
        df = pd.DataFrame({"": metrics_dict}).round(2)
        df = df.reindex([k for k in metrics_dict.keys()])
        print(df)


def main():
    """
    Command line interface for calculating metrics.

    Usage:
        1. Provide md_dirs, ensemble_dirs and names:

        `
        python analysis_script.py --md_dirs md_dir1 md_dir2 ... \
            --ensemble_dirs ensemble_dir1 ensemble_dir2 ... \
            --names name1 name2 ...
        `

        2. Provide md_dirs, ensemble_dirs and auto-detect names from ensemble_dirs:

        `
        python analysis_script.py --md_dirs md_dir1 md_dir2 ... \
            --ensemble_dirs ensemble_dir1 ensemble_dir2 ... 
        `

        3. Provide paths directly of the form:    

        `
        python analysis_script.py --paths \
            model1.pdb ref1.pdb ref_traj1.xtc ... ref_trajN1.xtc \
            model2.pdb ref2.pdb ref_traj1.xtc ... ref_trajN2.xtc ...
        `

        
        Additional arguments:

        ```
        --num_workers N : Number of parallel workers (default: 8)
        --metrics metric1 metric2 ... : Subset of metrics to calculate (default: all)
        --use_tqdm : Show progress bar
        --analysis_data_path path : Path to save raw analysis data (pickle file)
        --analysis_metrics_path path : Path to save metrics summary (CSV file)
        --print_metrics : Print metrics to console
        ```
                                    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_dirs', nargs='*', default=[])
    parser.add_argument('--ensemble_dirs', nargs='*', default=[])
    parser.add_argument('--names', nargs='*', default=[])
    parser.add_argument('--paths', nargs='*', default=[])
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--metrics', nargs='*', default=default_metrics)
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--analysis_data_path', type=str, default='')
    parser.add_argument('--analysis_metrics_path', type=str, default='')
    parser.add_argument('--print_metrics', action='store_true')
    args = parser.parse_args()

    # Either provide md_dirs, ensemble_dirs and names,
    # or provide paths where each path contains both md and ensemble pdbs.
    if len(args.paths) > 0:
        if len(args.md_dirs) > 0 or len(args.ensemble_dirs) > 0 or len(args.names) > 0:
            raise ValueError("Either provide --paths OR --md_dirs, --ensemble_dirs and --names.")
        args.md_dirs = None
        args.ensemble_dirs = None
        args.names = None
        paths = []
        # args.paths must be in the format:
        # model1.pdb, ref1.pdb, ref_traj1.xtc, ..., ref_trajN1.xtc,
        # model2.pdb, ref2.pdb, ref_traj1.xtc, ..., ref_trajN2.xtc,
        # ...
        paths.append([])
        for p in args.paths:
            if p.endswith('.pdb') and len(paths[-1]) < 2:
                # model ensemble pdb or reference pdb
                paths[-1].append(p)
            elif p.endswith('.pdb') and len(paths[-1]) >= 3:
                # start a new entry
                paths.append([p])
            elif p.endswith('.xtc'):
                # reference traj
                if len(paths[-1]) < 2:
                    raise ValueError("Reference traj provided before model ensemble pdb or reference pdb.")
                paths[-1].append(p)
            elif p.endswith('.pdb') and len(paths[-1]) == 2:
                raise ValueError("More than two pdbs provided before reference trajs.")
            
        if len(paths[-1]) == 0:
            paths = paths[:-1]

        if len(paths) == 0:
            raise ValueError("No valid paths provided.")
        
        if not paths[-1][-1].endswith('.xtc'):
            raise ValueError("Last entry in paths missing reference trajs.")
        
        for i in range(len(paths)):
            paths[i] = tuple(paths[i])

        names = None
        
    else:
        paths = None
        if len(args.names) == 0:
            # auto-detect names from ensemble_dirs. 
            # Include all pdbs and dirs in ensemble_dirs
            names = []
            for ensemble_dir in args.ensemble_dirs:
                names_ = [
                    f for f in os.listdir(ensemble_dir)
                    if f.endswith('.pdb') or os.path.isdir(os.path.join(ensemble_dir, f))
                ]
                names.extend(names_)
            names = list(set(names))
            print(f"Found {len(names)} generated ensembles to analyze:")
            print(names)
        else:
            names = args.names

        assert len(names) > 0, "No names provided for analysis."
    

    calc_metrics(
        names=names,
        md_dirs=args.md_dirs,
        ensemble_dirs=args.ensemble_dirs,
        paths=paths,
        num_workers=args.num_workers,
        metrics_subset=args.metrics,
        use_tqdm=not args.no_tqdm,
        analysis_data_path=args.analysis_data_path,
        analysis_metrics_path=args.analysis_metrics_path,
        print_metrics=args.print_metrics,
    )



if __name__ == "__main__":
    main()
