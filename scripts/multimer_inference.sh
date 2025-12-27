# define path to test data
THISDIR=$(dirname "$(readlink -f "$0")")
TEST_PDB_DIR="${THISDIR}/../test_data/test_pdbs"
set -e

bbflow_sample --input_path "${TEST_PDB_DIR}/multimer.pdb"\
    --output_path "${TEST_PDB_DIR}/multimer_samples.pdb"\
    --cuda_memory_GB 5 \
    --num_samples 10 \
    --tag bbflow-multimer-0.2
