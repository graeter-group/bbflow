# THIS SCRIPT USES A LIGHTWEIGHT EXAMPLE MODEL FOR DEMONSTRATION. TO OBTAIN GOOD ENSEMBLES, CHANGE THE MODEL TAG TO 'LATEST' BELOW.

# Example usage of the bbflow_sample command (run bbflow_sample -h for help)

# define path to test data
THISDIR=$(dirname "$(readlink -f "$0")")
TEST_PDB_DIR="${THISDIR}/../test_data/test_pdbs"
set -e


# EXAMPLE 1
# specify (several) input and output path for each equilibrium structure and ensemble
# the output_path may also end on .xtc, in this case, the ensemble will be saved in compressed format
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


# EXAMPLE 3
# Sample with fixed residues: keep all residues fixed except for residues 18-26 and 32-47
# (equivalent to keeping 1-17, 27-31, 48-end fixed).
# --keep_fixed accepts the same string format as the Python API:
#   '5-14,20,25-30;13-18'  =>  ';' separates chains, ',' separates parts within a chain
#   Prefix a chain block with '~' to negate (keep those residues flexible instead).
#   Indices are 1-based and relative to each chain.
# --fixed_residue_scaling 0.0 enforces the constraint exactly (larger values allow softer constraints).
bbflow_sample --input_path "${TEST_PDB_DIR}/equilibrium.pdb" \
    --output_path "${TEST_PDB_DIR}/equilibrium_fixed_samples.pdb" \
    --cuda_memory_GB 5 \
    --num_samples 50 \
    --tag bbflow-mini-0.1 \
    --keep_fixed "~18-26,32-47" \
    --fixed_residue_scaling 0.0