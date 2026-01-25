#!/bin/bash
# Protein Design Hub - Environment Setup Script

set -e

echo "========================================"
echo "  Protein Design Hub - Setup Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓ conda found${NC}"

# Create conda environment
ENV_NAME="protein_design_hub"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists${NC}"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment..."
    fi
fi

echo ""
echo "Creating conda environment with OpenStructure..."
conda env create -f environment.yaml -y || conda env update -f environment.yaml

echo ""
echo -e "${GREEN}✓ Conda environment created${NC}"

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install package in development mode
echo ""
echo "Installing protein_design_hub package..."
pip install -e ".[dev]"

echo ""
echo -e "${GREEN}✓ Package installed${NC}"

# Check GPU
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo -e "${YELLOW}Warning: PyTorch not available${NC}"

# Check OpenStructure
echo ""
echo "Checking OpenStructure..."
python -c "import ost; print(f'OpenStructure: OK')" 2>/dev/null || echo -e "${YELLOW}Warning: OpenStructure not available (some metrics may not work)${NC}"

# Summary
echo ""
echo "========================================"
echo -e "${GREEN}  Setup Complete!${NC}"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run the CLI:"
echo "  pdhub --help"
echo ""
echo "To start the web UI:"
echo "  pdhub web"
echo ""
echo "To install prediction tools:"
echo "  pdhub install --all"
echo ""
