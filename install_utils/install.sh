TORCH_VERSION=${1:-"2.6.0"}
CUDA_VERSION=${2:-"124"}

echo "Installing bbflow with torch $TORCH_VERSION and cuda $CUDA_VERSION"

# wait for 3 seconds in case the user wishes to cancel the installation
sleep 3

THISDIR=$(dirname "$(readlink -f "$0")")

pushd "${THISDIR}/../.."

# install torch
pip install torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cu$CUDA_VERSION
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html

# install pypi dependencies:
pip install -r bbflow/install_utils/requirements.txt

# install gafl from source:
# Note: this is a temporary solution until gafl is available on pypi
git clone https://github.com/hits-mli/gafl.git
pushd gafl
bash install_gatr.sh # Apply patches to gatr (needed for gafl)
pip install -e . # Install GAFL
popd

# make sure the provided version has not been changed in the previous steps:
pip install torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/cu$CUDA_VERSION
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html

# Finally, install bbflow:
cd bbflow
pip install -e . # Install bbflow

popd