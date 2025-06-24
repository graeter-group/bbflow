# Learning conformational ensembles of proteins with BBFlow

Structure-based conformational ensemble prediction with flow matching. The method is described in the paper [Learning conformational ensembles of proteins based on backbone geometry](https://arxiv.org/abs/2503.05738).

```
@article{wolf2025conformational,
      title={Learning conformational ensembles of proteins based on backbone geometry}, 
      author={Nicolas Wolf and Leif Seute and Vsevolod Viliuga and Simon Wagner and Jan Stühmer and Frauke Gräter},
      year={2025},
      eprint={2503.05738},
      archivePrefix={arXiv},
      journal={arXiv preprint arXiv:2503.05738},
}
```

Please cite the paper if you use the code.


This repository relies on the [GAFL](https://github.com/hits-mli/gafl) package and code from [FrameFlow](https://github.com/microsoft/protein-frame-flow).


# Usage

After installing the bbflow package, you can generate ensemble directly in python:

```python
from bbflow.deployment.bbflow import BBFlow
bbflow_sampler = BBFlow.from_tag('latest')
bbflow_sampler.sample(input_path='<path/to/equilibrium.pdb>', output_path='<path/to/output_ensemble.pdb>', num_samples=50)
```

or by using the command line interface:

```bash
bbflow_sample --input_path <path/to/equilibrium.pdb> --output_path <path/to/output_ensemble.pdb> --num_samples 50
```

For more details, see the example scripts at `scripts/inference_example.py` and `scripts/inference_example.sh`.

We also provide an [example notebook in Google Colab](https://colab.research.google.com/drive/11_xvYYOM5ckmY9PiKUCLcAAB_cIpfzG4?usp=sharing), for which no local installation is required.


# Installation

## TLDR

You can use our install script (here for torch version 2.6.0 and cuda 12.4), which esssentially executes the steps specified in section **pip** below:

```bash
git clone https://github.com/graeter-group/bbflow.git
conda create -n bbflow python=3.10 pip=23.2.1 -y
conda activate bbflow && bash bbflow/install_utils/install_via_pip.sh 2.6.0 124
```

Verify your installation by running our example script or `pytest`:

```bash
bash bbflow/scripts/inference_example.sh
```

## pip

Optional: Create a virtual environment, e.g. with conda and install pip23.2.1:

```bash
conda create -n bbflow python=3.10 pip=23.2.1 -y
conda activate bbflow
```

Install the dependencies from the requirements file:

```bash
git clone https://github.com/graeter-group/bbflow.git
pip install -r bbflow/install_utils/requirements.txt

# BBFlow builds on top of the GAFL package, which is installed from source:
git clone https://github.com/hits-mli/gafl.git
cd gafl
bash install_gatr.sh # Apply patches to gatr (needed for gafl)
pip install -e . # Install GAFL
cd ..

# Finally, install bbflow with pip:
cd bbflow
pip install -e .
```

Install torch with a suitable cuda version, e.g.

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

where you can replace cu124 by your cuda version, e.g. cu118 or cu121.

## conda

BBFlow relies on the [GAFL](https://github.com/hits-mli/gafl) package, which can be installed from GitHub as shown below. The dependencies besides GAFL are listed in `install_utils/environment.yaml`, we also provide a minimal environment in `install_utils/minimal_env.yaml`, where it is easier to change torch/cuda versions.

```bash
# download bbflow:
git clone https://github.com/graeter-group/bbflow.git
# create env with dependencies:
conda env create -f bbflow/install_utils/minimal_env.yaml
conda activate bbflow

# install gafl:
git clone https://github.com/hits-mli/gafl.git
cd gafl
bash install_gatr.sh # Apply patches to gatr (needed for gafl)
pip install -e .
cd ..

# install bbflow:
cd bbflow
pip install -e .
```

## Common installation issues

Problems with torch_scatter can usually be resolved by uninstalling and re-installing it via pip for the correct torch and cuda version, e.g. `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu124.html` for torch 2.0.0 and cuda 12.4.
