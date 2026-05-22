#!/bin/bash
# Test script for the fix-residue CLI arguments (--keep_fixed, --fixed_residue_scaling)

set -e

THISDIR=$(dirname "$(readlink -f "$0")")
TEST_PDB_DIR="${THISDIR}/../test_data/test_pdbs"
OUTPUT_DIR="${THISDIR}/../test_data/test_fix_residue"
mkdir -p "${OUTPUT_DIR}"

echo "=== Test: fix residue via CLI --keep_fixed and --fixed_residue_scaling ==="

# Keep all residues fixed except 18-26 and 32-47 (using negation prefix ~)
bbflow_sample \
    --input_path "${TEST_PDB_DIR}/equilibrium.pdb" \
    --output_path "${OUTPUT_DIR}/fixed_residue_test.pdb" \
    --num_samples 10 \
    --tag bbflow-mini-0.1 \
    --keep_fixed "~18-26,32-47" \
    --fixed_residue_scaling 0.0 \
    --device cuda

echo ""
echo "Output written to ${OUTPUT_DIR}/fixed_residue_test.pdb"
echo "Also check for superposed output: ${OUTPUT_DIR}/fixed_residue_test_superposed.pdb"
echo "=== Done ==="
