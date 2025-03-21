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


# Installation

BBFlow relies on the [GAFL](https://github.com/hits-mli/gafl) package, which can be installed from GitHub as shown below. The dependencies besides GAFL are listed in `environment.yaml`.

```bash
# create env with dependencies:
conda env create -f environment.yaml
conda activate bbflow
# install gafl:
git clone git@github.com:hits-mli/gafl.git ../gafl
pushd ../gafl
bash install_gatr.sh # Apply patches to gatr (needed for gafl)
pip install -e .
# install bbflow:
popd
pip install -e .
```

# Usage

After installing the bbflow package, you can generate ensemble states by three lines of code:

```python
from bbflow.deployment.bbflow import BBFlow
bbflow_sampler = BBFlow.from_tag('latest')
bbflow_sampler.sample(input_path='<path/to/equilibrium.pdb>', output_path='<path/to/output_ensemble.pdb>', n_samples=50)
```

For more details, see the example script in `scripts/inference_example.py`.
