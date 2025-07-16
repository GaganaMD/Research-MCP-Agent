# scripts/setup_environment.sh
#!/bin/bash

echo "Setting up MCP Research Agent Environment..."

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: NVIDIA GPU not detected. CPU-only mode will be used."
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/documents
mkdir -p data/training
mkdir -p data/embeddings
mkdir -p logs
mkdir -p models

# Set up pre-commit hooks
pre-commit install

echo "Environment setup complete!"
echo "Activate with: source venv/bin/activate"