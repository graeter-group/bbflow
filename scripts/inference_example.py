# THIS SCRIPT USES A LIGHTWEIGHT EXAMPLE MODEL FOR DEMONSTRATION. TO OBTAIN GOOD ENSEMBLES, CHANGE THE MODEL TAG TO 'LATEST' BELOW.

#%%

from bbflow.deployment.bbflow import BBFlow
from pathlib import Path

rootdir = Path(__file__).parent.parent

# Download a small test model into ./models/bbflow-mini-0.1, 
bbflow_sampler = BBFlow.from_tag('bbflow-mini-0.1') # use 'latest' to download a heavier, more accurate model

# Alternatively, you can specify a ckpt path directly
# ckpt_path = './ckpt/bbflow.ckpt'
# bbflow_sampler = BBFlow(ckpt_path)

# %%
# Sample conformations from the protein specified
# by the PDB file of the equilibrium structure
pdb_path = rootdir/'test_data/test_pdbs/equilibrium.pdb'

#%%
# Sample 50 conformations and save them to the output path 
# with a fixed batch size of 10
bbflow_sampler.sample(
    input_path=pdb_path, output_path=rootdir/'test_data/test_pdbs/ensemble.pdb', 
    num_samples=50, batch_size=10
)

#%%
# Sample 50 conformations and save them to the output path where
# the batch size is estimated based on the VRAM of the GPU
bbflow_sampler.sample(
    input_path=pdb_path, output_path=rootdir/'test_data/test_pdbs/ensemble.pdb', 
    num_samples=50, cuda_memory_GB=6
)


#%%
# Sample 50 conformations, but keep part of the structure fixed

# 'keep_fixed' defines which residues to keep fixed during sampling.
# 'keep_fixed' can be either a boolean mask (numpy array or torch tensor) of shape (num_residues,) 
# where True indicates that the residue should be kept fixed, 
# or a string of the form '5-14,20,25-30;~13-18' where ';' separates chains and ',' separates parts for each chain. 
# Each part can be either a single index (e.g. '20') or a range (e.g. '5-14') where the indices are inclusive.
# If a chain starts with '~', the mask for the whole chain is negated (i.e. those residues will be flexible instead of fixed).
# The indices are 1-based and relative to each chain (e.g., residue 1 is the first residue of each chain).

# 'fixed_residue_scaling' defines how strongly the fixed residues are kept fixed.
# During inference, the prediction for the fixed part is replaced by a geodesic interpolation between the predicted structure
# and the input structure, using the interpolation factor beta_t = 1 - fixed_residue_scaling * t, where t is the flow matching time.
# When fixed_residue_scaling=0, the fixed residues are moved exactly to the input structure
# fixed_residue_scaling>0 results in a softer constraint

bbflow_sampler.sample(
    input_path=pdb_path, output_path=rootdir/'test_data/test_pdbs/ensemble.pdb', 
    num_samples=50, cuda_memory_GB=6,
    keep_fixed="~18-26,32-47", # equivalent to "1-17,27-31,48-end": keep all residues fixed except for 18-26 and 32-47 (inclusive)
    fixed_residue_scaling=0.0 # keep the fixed residues completely fixed
)


# %%
# Analyze the generated ensemble against the reference MD trajectory
from scripts.analysis_script import calc_metrics

calc_metrics(
    paths=[
        (
            "test_data/test_pdbs/ensemble.pdb",
            "test_data/test_pdbs/equilibrium.pdb",
            "test_data/test_pdbs/MD_trajectory.xtc",
        )
    ],
    print_metrics=True,
    analysis_data_path="test_data/test_pdbs/analysis_data.pkl",
    analysis_metrics_path="test_data/test_pdbs/metrics_summary.csv",
    num_workers=1,
)

