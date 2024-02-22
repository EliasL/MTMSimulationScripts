import subprocess
from pathlib import Path

# Path to your .scpt file
script_path = f'{Path(__file__).resolve().parent}/startMonitoring.scpt'

# Running the AppleScript
process = subprocess.run(['osascript', script_path], capture_output=True, text=True)

# Getting the output
stdout = process.stdout
stderr = process.stderr

# Check if there was an error
if process.returncode == 0:
    print(f'Script executed successfully: {stdout}')
else:
    print(f'Error executing script: {stderr}')
