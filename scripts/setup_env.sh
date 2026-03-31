#!/bin/bash
# Setup script for Video Dubbing Vietnamese
# Installs CUDA toolkit, system dependencies, and Python packages

set -e

echo "🔧 Setting up Video Dubbing Vietnamese environment..."

# System dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    rubberband-cli \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/raw
mkdir -p data/reference_audio
mkdir -p data/processed
mkdir -p data/outputs
mkdir -p data/eval_set
mkdir -p models/qwen2-vl-7b-q4
mkdir -p models/glm-ocr
mkdir -p models/vitts
mkdir -p docs/eval_results

# Check CUDA
echo "🔍 Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ CUDA GPU detected"
else
    echo "⚠️  No NVIDIA GPU detected. Local models will run on CPU (slow)."
fi

# Copy .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env from .env.example — please edit with your API keys"
fi

echo ""
echo "✅ Setup complete!"
echo "   1. Edit .env with your GEMINI_API_KEY"
echo "   2. Download models: python scripts/download_models.py"
echo "   3. Run: python scripts/run_pipeline.py --help"
