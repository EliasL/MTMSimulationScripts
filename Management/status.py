import subprocess
from pathlib import Path

# Path to your .scpt file
script_path = f'{Path(__file__).resolve().parent}/startMonitoring.scpt'

# Running the AppleScript
process = subprocess.run(['osascript', script_path], capture_output=True, text=True)
# Getting the output
stderr = process.stderr

# Check if there was an error
if process.returncode != 0:
    print(f'Error executing script: {stderr}')
