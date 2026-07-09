#!/bin/bash
# Orchard installation script - does everything for you

set -e  # Exit on any error

echo "============================================"
echo "Orchard Installation Script"
echo "============================================"

# 1. Clone the repository
echo "Step 1: Cloning Orchard repository..."
if [ ! -d "orchard" ]; then
    git clone https://github.com/morrislab/orchard.git
    cd orchard
else
    echo "Orchard directory already exists, using it..."
    cd orchard
fi

ORCH_DIR=$(pwd)
echo "Orchard directory: $ORCH_DIR"

# 2. Create conda environment
echo ""
echo "Step 2: Creating conda environment..."
conda create -y --name orchard python=3.10

# 3. Set up environment variables in conda
echo ""
echo "Step 3: Setting up environment variables..."
CONDA_ACTIVATE_ENV_VARS=$HOME/.conda/envs/orchard/etc/conda/activate.d/env_vars.sh
CONDA_DEACTIVATE_ENV_VARS=$HOME/.conda/envs/orchard/etc/conda/deactivate.d/env_vars.sh

mkdir -p $HOME/.conda/envs/orchard/etc/conda/activate.d
mkdir -p $HOME/.conda/envs/orchard/etc/conda/deactivate.d

touch $CONDA_ACTIVATE_ENV_VARS
touch $CONDA_DEACTIVATE_ENV_VARS

echo "" >> $CONDA_ACTIVATE_ENV_VARS
echo "# Set Environment Variables" >> $CONDA_ACTIVATE_ENV_VARS
echo "export ORCH_DIR=$ORCH_DIR" >> $CONDA_ACTIVATE_ENV_VARS

echo "" >> $CONDA_DEACTIVATE_ENV_VARS
echo "# Deactivate Environment Variables" >> $CONDA_DEACTIVATE_ENV_VARS
echo "unset ORCH_DIR" >> $CONDA_DEACTIVATE_ENV_VARS

# 4. Activate environment and install dependencies
echo ""
echo "Step 4: Activating environment and installing Python dependencies..."
source activate orchard

# 5. Install Python requirements
echo ""
echo "Step 5: Installing Python packages..."
python -m pip install -r requirements.txt

# 6. Build projectppm
echo ""
echo "Step 6: Building projectppm (required dependency)..."
cd $ORCH_DIR/lib
if [ ! -d "projectppm" ]; then
    git clone https://github.com/ethanumn/projectppm
fi
cd projectppm
bash make.sh
cd $ORCH_DIR

# 7. Done!
echo ""
echo "============================================"
echo "✓ Installation Complete!"
echo "============================================"
echo ""
echo "To use Orchard:"
echo "  1. Activate the environment:"
echo "     conda activate orchard"
echo ""
echo "  2. Test with example data:"
echo "     python3 \$ORCH_DIR/bin/orchard examples/example1/example1.ssm examples/example1/example1.params.json examples/example1/example1.orchard.npz --seed=123"
echo ""
echo "Environment variable \$ORCH_DIR is automatically set when you activate the environment."
echo ""
