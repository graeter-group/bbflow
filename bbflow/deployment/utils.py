# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import numpy as np
from pathlib import Path
from typing import Union
import requests
from tqdm.auto import tqdm


LATEST_TAG = 'bbflow-0.1'
CKPT_URLS = {
    'bbflow-0.1': 'https://keeper.mpdl.mpg.de/f/bee3c06a20b94ee2ac92/?dl=1',
}
CONFIG_URLS = {
    'bbflow-0.1': 'https://keeper.mpdl.mpg.de/f/7ed49e41d7c444569a11/?dl=1',
}


def estimate_max_batchsize(n_res, memory_GB=40):
    """
    Estimate the maximum batch size that can be used for sampling. Hard-coded from empiric experiments. We found a dependency that is inversely proportional to the number of residues in the protein.
    """
    if not isinstance(n_res, np.ndarray):
        n_res = np.array([n_res])
    A = 5e4
    B = 40
    batchsize = A/(n_res+B)**2 * memory_GB
    batchsize = np.floor(batchsize)
    ones = np.ones_like(batchsize)
    out = np.max(np.stack([batchsize, ones], axis=0), axis=0)
    if out.shape == (1,):
        return int(out[0])
    return out.astype(int)


# overwrite args that are specified in cfg:
def recursive_update(cfg:dict, cfg_:dict):
    for key in cfg.keys():
        if key in cfg_.keys():
            if isinstance(cfg[key], dict):
                recursive_update(cfg[key], cfg_[key])
            else:
                cfg_[key] = cfg[key]

def get_root_dir()->Path:
    """
    Get the root directory of the package.
    """
    return Path(__file__).parent.parent.parent


def ckpt_path_from_tag(tag:str='latest'):
    """
    Get the path to the checkpoint assumed to be located at root_dir/models/tag/*.ckpt. Checks existence and uniqueness of the checkpoint file and existence of the config file.
    """

    if tag == 'latest':
        tag = LATEST_TAG

    root_dir = get_root_dir()
    ckpt_dir = root_dir / 'models' / tag

    if not ckpt_dir.exists():
        if not tag in CKPT_URLS.keys():
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} not found and {tag} not found in the hard-coded URLs for downloading.")
        else:
            ckpt_path = download_model(tag)
            return ckpt_path
    
    ckpt_files = list(ckpt_dir.glob('*.ckpt'))
    if len(ckpt_files) == 0:
        if not tag in CKPT_URLS.keys():
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir} and {tag} not found in the hard-coded URLs for downloading.")
        else:
            ckpt_path = download_model(tag)
            return ckpt_path
    elif len(ckpt_files) > 1:
        raise FileNotFoundError(f"Multiple checkpoint files found in {ckpt_dir}.")
    if not (ckpt_dir/'config.yaml').exists():
        raise FileNotFoundError(f"No config file found in {ckpt_dir}.")

    return ckpt_files[0]

def _download_file(url:str, target_path:Path, progress_bar:bool=True):
    # Start the download
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure the request was successful

    if progress_bar:
        # Get the total file size from headers
        total_size = int(response.headers.get('content-length', 0))

        # Initialize the progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True) as t:
            with open(target_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024*2**2):
                    file.write(chunk)
                    t.update(len(chunk))
    else:
        with open(target_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024*2**2):
                file.write(chunk)

    assert target_path.exists(), f"Download failed, file not found at {target_path}"


def download_model(tag:str='latest'):
    """
    Download the model checkpoint and config file from the hard-coded URLS.
    """
    if tag == 'latest':
        tag = LATEST_TAG

    root_dir = get_root_dir()
    ckpt_dir = root_dir / 'models' / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_url = CKPT_URLS[tag]
    config_url = CONFIG_URLS[tag]

    ckpt_path = ckpt_dir / f'{tag}.ckpt'
    config_path = ckpt_dir / 'config.yaml'

    _download_file(config_url, config_path, progress_bar=False)
    _download_file(ckpt_url, ckpt_path, progress_bar=True)

    return ckpt_path