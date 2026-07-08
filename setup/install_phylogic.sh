#!/usr/bin/env bash
# setup/install_phylogic.sh
#
# Run ONCE after creating the exome_phylogic conda environment.
# Clones PhylogicNDT into a local directory and installs its Python
# dependencies into the active conda environment.
#
# Usage:
#   conda activate exome_phylogic
#   bash setup/install_phylogic.sh [--dest /path/to/tools]
#
# After running, set phylogic.phylogicndt_dir in config.yaml to the
# cloned directory path printed at the end of this script.

set -euo pipefail

DEST="${1:-$(pwd)/tools}"
if [[ "$1" == "--dest" ]]; then DEST="$2"; fi

PHYLOGIC_DIR="${DEST}/PhylogicNDT"

echo "=== Installing PhylogicNDT ==="
echo "Destination: ${PHYLOGIC_DIR}"

mkdir -p "${DEST}"

if [[ -d "${PHYLOGIC_DIR}/.git" ]]; then
    echo "PhylogicNDT already cloned — pulling latest..."
    git -C "${PHYLOGIC_DIR}" pull
else
    git clone https://github.com/broadinstitute/PhylogicNDT.git "${PHYLOGIC_DIR}"
fi

# Install Python dependencies from the repo's requirements file
if [[ -f "${PHYLOGIC_DIR}/requirements.txt" ]]; then
    pip install -r "${PHYLOGIC_DIR}/requirements.txt"
elif [[ -f "${PHYLOGIC_DIR}/req" ]]; then
    pip install -r "${PHYLOGIC_DIR}/req"
else
    # Fallback: install known dependencies from README
    pip install "pandas>=1.0" "scipy>=1.0" "matplotlib>=2.0"
fi

# Make the main script executable
chmod +x "${PHYLOGIC_DIR}/PhylogicNDT.py"

echo ""
echo "=== PhylogicNDT installed successfully ==="
echo ""
echo "Add this to config.yaml:"
echo "  phylogic:"
echo "    phylogicndt_dir: ${PHYLOGIC_DIR}"
echo "    n_iter: 1000"
echo ""
echo "Offline HPC: copy ${PHYLOGIC_DIR} to the cluster and set the path."
