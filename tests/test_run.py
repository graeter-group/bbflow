import pytest
from bbflow.deployment.bbflow import BBFlow
from pathlib import Path

@pytest.fixture(scope="module")
def bbflow_sampler():
    """
    Fixture to set up the BBFlow model for testing.
    This will download the model.
    """
    bbflow_sampler = BBFlow.from_tag('bbflow-mini-0.1', force_download=True)
    
    return bbflow_sampler

def test_run(bbflow_sampler):
    """
    Run a mini model to test the BBFlow functionality. This is essentially the inference_example.py script for a short backbone.
    """
    rootdir = Path(__file__).parent.parent
    # Load the BBFlow model

    pdb_path = rootdir/'test_data/test_pdbs/short_equilibrium.pdb'
    outpath = rootdir/'test_data/test_pdbs/test_samples.pdb'

    if outpath.exists():
        outpath.unlink()

    bbflow_sampler.sample(
        input_path=pdb_path, output_path=outpath,
        num_samples=2, batch_size=1
    )