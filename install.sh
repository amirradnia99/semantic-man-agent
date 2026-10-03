#!/bin/bash
# =============================================================================
# Linux Command Knowledge & Retrieval Engine  --  installer
# =============================================================================
set -e

echo "🐧 Linux Command Knowledge & Retrieval Engine"
echo "=============================================="

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)'; then
    echo "❌ Python 3.8+ required (found $(python3 --version 2>&1))"
    exit 1
fi
echo "✅ Python $(python3 --version 2>&1 | awk '{print $2}')"

for t in groff man; do
    if ! command -v "$t" >/dev/null 2>&1; then
        echo "❌ Missing: $t"
        echo "   Install with: sudo apt install -y groff man-db manpages manpages-dev"
        exit 1
    fi
done
echo "✅ groff and man"

if ! python3 -c "import sqlite3; sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)')" 2>/dev/null; then
    echo "❌ Python's sqlite3 lacks FTS5. Use a distro Python (not a stripped build)."
    exit 1
fi
echo "✅ sqlite3 with FTS5"

if [ ! -d venv ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

chmod +x semantic_man.py

mkdir -p "$HOME/.local/bin"
ln -sf "$(pwd)/semantic_man.py" "$HOME/.local/bin/semantic-man" 2>/dev/null || true

echo ""
echo "✅ Installation complete."
echo ""
echo "Next steps (activate venv first):"
echo "  source venv/bin/activate"
echo "  python semantic_man.py doctor"
echo "  python semantic_man.py index --sections all"
echo "  python semantic_man.py benchmark --held-out"
echo "  python semantic_man.py agent"
