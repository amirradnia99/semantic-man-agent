#!/bin/bash
# Semantic Man-Page Agent Installer

set -e

echo "🐧 Semantic Man-Page Agent Installer"
echo "====================================="

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$(echo "$python_version < 3.8" | bc)" -eq 1 ]; then
    echo "❌ Python 3.8+ required (found $python_version)"
    exit 1
fi
echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install sentence-transformers chromadb numpy rich

# Make script executable
chmod +x semantic_man.py

# Create symlink
mkdir -p ~/.local/bin
ln -sf "$(pwd)/semantic_man.py" ~/.local/bin/semantic-man

echo ""
echo "✅ Installation complete!"
echo ""
echo "To use:"
echo "  semantic-man doctor      # Check system health"
echo "  semantic-man index       # Build the index"
echo "  semantic-man agent       # Start interactive mode"
echo "  semantic-man search '...' # Quick search"
echo ""
echo "Make sure ~/.local/bin is in your PATH:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
