# Example usage of the bbflow_sample command (run bbflow_sample -h for help)

# define path to test data
THISDIR=$(dirname "$(readlink -f "$0")")
TEST_PDB_DIR="${THISDIR}/../test_data/test_pdbs"
set -e


# EXAMPLE 1
# specify (several) input and output path for each equilibrium structure and ensemble
# the output_path may also end on .xtc, in this case, a the trajectory will be saved in compressed format
bbflow_sample --input_path "${TEST_PDB_DIR}/equilibrium.pdb" "${TEST_PDB_DIR}/short_equilibrium.pdb" \
    --output_path "${TEST_PDB_DIR}/equilibrium_samples.pdb" "${TEST_PDB_DIR}/short_equilibrium_samples.pdb" \
    --cuda_memory_GB 5 \
    --num_samples 100 \
    --tag bbflow-mini-0.1 # small test model, use --tag latest for more accurate ensembles


# EXAMPLE 2
# you can also pass an input and output directory to generate ensembles for all pdb files in that directory
# to demonstrate this, we create a seperate directory with example inputs:
mkdir -p "${TEST_PDB_DIR}/example_inputs"
cp "${TEST_PDB_DIR}/equilibrium.pdb" "${TEST_PDB_DIR}/example_inputs/equilibrium.pdb"
cp "${TEST_PDB_DIR}/short_equilibrium.pdb" "${TEST_PDB_DIR}/example_inputs/short_equilibrium.pdb"

bbflow_sample --input_dir "${TEST_PDB_DIR}/example_inputs" \
    --output_dir "${TEST_PDB_DIR}/example_outputs" \
    --cuda_memory_GB 5 \
    --num_samples 100 \
    --tag bbflow-mini-0.1 # small test model, use --tag latest for more accurate ensembles