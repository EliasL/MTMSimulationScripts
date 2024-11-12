#!/bin/bash

# Get the current project directory
PROJECT_DIR=$(pwd)

# Name of the virtual environment directory
VENV_DIR="$PROJECT_DIR/venv"

# Check if virtualenv exists and create it if not
echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"


# Make sure that the Python binary is rebuilt for the current system architecture
rm -f "$VENV_DIR/bin/python*"  # Remove old Python binaries

# Recreate the Python binaries
python3 -m venv --upgrade "$VENV_DIR"

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Skip installing packages if there is no internet
# You can add a check to see if you're on the cluster, 
# or assume it doesn't have internet

if ping -c 1 google.com &> /dev/null; then
    # If there is an internet connection, try installing dependencies
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        echo "Installing required libraries from requirements.txt..."
        pip3 install -r "$PROJECT_DIR/requirements.txt"
    else
        echo "No requirements.txt found. Skipping package installation."
    fi
else
    echo "No internet connection detected. Skipping package installation."
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