TORCH_VERSION=${1:-"2.6.0"}
CUDA_VERSION=${2:-"124"}
PYTHON_VERSION=${3:-"None"}

BBFLOW_DIR="bbflow-dev"

SUPPORTED_VERSIONS=("3.10" "3.12" "None")
# assert that python version is either 3.10, 3.12 or None:
if [[ ! " ${SUPPORTED_VERSIONS[*]} " =~ " ${PYTHON_VERSION} " ]]; then
    echo "Error: Unsupported python version $PYTHON_VERSION. Supported versions are: ${SUPPORTED_VERSIONS[*]}"
    exit 1
fi

echo "Installing bbflow with torch $TORCH_VERSION and cuda $CUDA_VERSION"

# wait for 3 seconds in case the user wishes to cancel the installation
sleep 3

THISDIR=$(dirname "$(readlink -f "$0")")

pushd "${THISDIR}/../.."

# install torch
pip install torch==$TORCH_VERSION torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu$CUDA_VERSION
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html

# add the torch version to the requirements file to make sure it is not overwritten
if [[ "$PYTHON_VERSION" != "None" ]]; then
    REQUIREMENTS_FILE="requirements_${PYTHON_VERSION//./}.txt"
else
    REQUIREMENTS_FILE="requirements.txt"
fi

cp $BBFLOW_DIR/install_utils/$REQUIREMENTS_FILE $BBFLOW_DIR/install_utils/tmp_requirements.txt
echo -e "\ntorch==$TORCH_VERSION" >> $BBFLOW_DIR/install_utils/tmp_requirements.txt

# install pypi dependencies:
pip install -r $BBFLOW_DIR/install_utils/tmp_requirements.txt

rm $BBFLOW_DIR/install_utils/tmp_requirements.txt

# install gafl from source:
# Note: this is a temporary solution until gafl is available on pypi
git clone https://github.com/hits-mli/gafl.git
pushd gafl
bash install_gatr.sh # Apply patches to gatr (needed for gafl)
pip install -e . # Install GAFL
popd

# Finally, install bbflow:
cd $BBFLOW_DIR
pip install -e . # Install bbflow

popd