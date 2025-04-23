# Example usage of the bbflow_sample command (run bbflow_sample -h for help)

THISDIR=$(dirname "$(readlink -f "$0")")

TEST_PDB_DIR="${THISDIR}/../test_data/test_pdbs"

bbflow_sample --input_path "${TEST_PDB_DIR}/equilibrium.pdb" "${TEST_PDB_DIR}/short_equilibrium.pdb" \
    --output_path "${TEST_PDB_DIR}/equilibrium_samples.pdb" "${TEST_PDB_DIR}/short_equilibrium_samples.pdb" \
    --cuda_memory_GB 5 \
    --num_samples 100 \
    --tag bbflow-mini-0.1 # small test model, use --tag latest for more accurate ensembles