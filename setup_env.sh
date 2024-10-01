#!/bin/bash

# Get the current project directory (assumes the script is in the root directory)
PROJECT_DIR=$(pwd)

# Name of the virtual environment directory (you can change this)
VENV_DIR="$PROJECT_DIR/venv"

# Check if virtualenv exists and create it if not
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install required libraries if requirements.txt is found
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "Installing required libraries from requirements.txt..."
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo "No requirements.txt found. Skipping package installation."
fi

# Update VSCode settings to use the created virtual environment
VSCODE_SETTINGS_DIR="$PROJECT_DIR/.vscode"
VSCODE_SETTINGS_FILE="$VSCODE_SETTINGS_DIR/settings.json"

if [ ! -d "$VSCODE_SETTINGS_DIR" ]; then
    mkdir "$VSCODE_SETTINGS_DIR"
fi

# Write to the settings.json file
echo "Updating VSCode workspace settings to use the virtual environment..."
cat > "$VSCODE_SETTINGS_FILE" << EOL
{
    "python.pythonPath": "$VENV_DIR/bin/python3"
}
EOL

echo "Environment setup is complete. The virtual environment is activated."