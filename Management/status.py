import subprocess
from pathlib import Path
import re

# Path to your .scpt file
script_path = f"{Path(__file__).resolve().parent}/startMonitoring.scpt"

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Read the AppleScript file
with open(script_path, "r") as file:
    applescript = file.read()


# Define a function to replace paths
def replace_path(match):
    # Extract the specific script name from the match
    script_name = match.group(0).split("Management/")[1]
    return f"python {current_dir}/{script_name}"


# Replace the paths in the AppleScript
pattern = r"python .*?/Management/.*"
applescript = re.sub(pattern, replace_path, applescript)

# Write the modified AppleScript to a temporary file
temp_script_path = current_dir / "temp_startMonitoring.scpt"
with open(temp_script_path, "w") as file:
    file.write(applescript)

# Running the AppleScript
process = subprocess.run(
    ["osascript", temp_script_path], capture_output=True, text=True
)
# Getting the output
stderr = process.stderr

# Check if there was an error
if process.returncode != 0:
    print(f"Error executing script: {stderr}")
else:
    print("Script executed successfully")

# Clean up the temporary file
temp_script_path.unlink()
