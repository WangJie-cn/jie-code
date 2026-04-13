#!/usr/bin/env bash
set -euo pipefail

echo "=== jie-code installer ==="
echo ""

# Check Python version
PYTHON="${PYTHON:-python3}"
PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "Error: Python 3.10+ required (found $PY_VERSION)"
    exit 1
fi
echo "Python $PY_VERSION OK"

# Install package
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing jie-code from $SCRIPT_DIR..."
$PYTHON -m pip install -e "$SCRIPT_DIR" --quiet

# Create config directory
JIE_HOME="${JIE_CONFIG_HOME:-$HOME/.jie}"
mkdir -p "$JIE_HOME"

# If ~/.claude exists but ~/.jie doesn't have skills/agents/commands,
# create symlinks for ecosystem compatibility
CLAUDE_HOME="$HOME/.claude"
if [ -d "$CLAUDE_HOME" ]; then
    for subdir in skills agents commands mcp-servers; do
        src="$CLAUDE_HOME/$subdir"
        dst="$JIE_HOME/$subdir"
        if [ -d "$src" ] && [ ! -e "$dst" ]; then
            ln -s "$src" "$dst"
            echo "Linked $dst -> $src"
        fi
    done
    # Link plugins
    src="$CLAUDE_HOME/plugins"
    dst="$JIE_HOME/plugins"
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
        echo "Linked $dst -> $src"
    fi
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Commands available:"
echo "  jie-code  - Full command"
echo "  jc        - Short alias"
echo ""
echo "Quick start:"
echo "  jc agent --profile anthropic-proxy -p 'hello'    # Claude API"
echo "  jc agent --profile local-qwen -p 'hello'         # Local model"
echo "  jc agent --profile ollama -p 'hello'              # Ollama"
echo ""
echo "Config: $JIE_HOME"
