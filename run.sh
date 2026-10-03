#!/usr/bin/env bash
# =============================================================================
# run.sh -- self-bootstrapping launcher for semantic-man-agent
#
# Usage:
#   ./run.sh "how do I compress a folder"
#   ./run.sh search "list open network ports" --verbose
#   ./run.sh agent
#   ./run.sh doctor
#
# On first run it:
#   - creates a local venv if missing
#   - pip installs dependencies
#   - builds the index if missing (asks first)
# =============================================================================
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="$HERE/venv"
PY="$VENV/bin/python"
CACHE="${SEMANTIC_MAN_CACHE:-$HOME/.cache/semantic-man}"

if [ ! -x "$PY" ]; then
    echo "==> First run: creating virtual environment ..."
    python3 -m venv "$VENV"
    "$PY" -m pip install --quiet --upgrade pip
    echo "==> Installing dependencies (this may take a few minutes) ..."
    "$PY" -m pip install --quiet -r "$HERE/requirements.txt"
fi

if [ ! -f "$CACHE/manifest.json" ]; then
    echo "==> No local index found under $CACHE"
    echo "    Building the index parses your system man pages and can take"
    echo "    30-60 minutes on a typical machine."
    read -r -p "    Build the index now? [y/N] " ans
    case "$ans" in
        [Yy]|[Yy][Ee][Ss])
            "$PY" "$HERE/semantic_man.py" index --sections all
            ;;
        *)
            echo "    Skipping. Build it later with:"
            echo "      $PY $HERE/semantic_man.py index --sections all"
            ;;
    esac
fi

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo
    echo "Examples:"
    echo "  $0 \"how do I compress a folder\""
    echo "  $0 search \"list open network ports\" --verbose"
    echo "  $0 agent"
    echo "  $0 doctor"
    exit 0
fi

FIRST="$1"
case "$FIRST" in
    index|search|agent|explain|troubleshoot|stats|doctor|benchmark|clear|version|--help|-h)
        exec "$PY" "$HERE/semantic_man.py" "$@"
        ;;
    *)
        exec "$PY" "$HERE/semantic_man.py" search "$@"
        ;;
esac
