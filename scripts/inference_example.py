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


# %%
