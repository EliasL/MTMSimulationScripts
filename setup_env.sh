#!/bin/bash

# Get the current project directory
PROJECT_DIR=$(pwd)

# Ensure ~/.local/bin is on PATH so uv can be found immediately after install
export PATH="$HOME/.local/bin:$PATH"

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create uv virtual environment
echo "Creating uv virtual environment..."
uv venv --python 3.11

# Determine uv-created venv path
VENV_DIR="$PROJECT_DIR/.venv"

# Install dependencies using uv pip
uv pip install -r "$PROJECT_DIR/requirements.txt"

# Install local packages in editable mode (auto-discover)
mapfile -t LOCAL_PACKAGES < <(
    find "$PROJECT_DIR" \
        -path "$PROJECT_DIR/.venv" -prune -o \
        -path "$PROJECT_DIR/.git" -prune -o \
        -path "*/__pycache__" -prune -o \
        -path "*/build" -prune -o \
        -path "*/dist" -prune -o \
        \( -name setup.py -o -name pyproject.toml \) -print \
    | xargs -I{} dirname "{}" \
    | sort -u
)

for pkg in "${LOCAL_PACKAGES[@]}"; do
    echo "Installing editable package: $pkg"
    uv pip install -e "$pkg"
done

# Update VSCode settings to use uv environment interpreter
VSCODE_SETTINGS_DIR="$PROJECT_DIR/.vscode"
VSCODE_SETTINGS_FILE="$VSCODE_SETTINGS_DIR/settings.json"

if [ ! -d "$VSCODE_SETTINGS_DIR" ]; then
    mkdir "$VSCODE_SETTINGS_DIR"
fi

echo "Updating VSCode workspace settings to use the uv virtual environment..."
cat > "$VSCODE_SETTINGS_FILE" << EOL
{
    "python.defaultInterpreterPath": "$VENV_DIR/bin/python"
}
EOL

echo "Environment setup is complete. Use 'uv run' or activate via 'source .venv/bin/activate'."
