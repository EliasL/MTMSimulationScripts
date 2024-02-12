from pathlib import Path
import subprocess
import time
import os

class SimulationManager:
    
    def __init__(self, configObj, outputPath=None, debugBuild=False, useProfiling=False):
        self.configObj = configObj
        self.outputPath = findOutputPath() if outputPath is None else outputPath

        self.useProfiling = useProfiling        
        self.project_path = str(Path(__file__).resolve().parent.parent.parent)
        # Change the working directory
        os.chdir(self.project_path)

        # Build folder
        self.debugBuild = debugBuild
        self.release_build_folder = "build-release/"
        self.profile_build_folder = "build/"
        self.build_folder = self.profile_build_folder if debugBuild else self.release_build_folder
        run_command(f"mkdir -p {self.build_folder}")
        # Build path
        self.build_path = os.path.join(self.project_path, self.build_folder)
     
        # I think it is better to always use release
        build_type = "Release"#"Debug" if self.useProfiling else "Release"
        self.build_command = f"cd {self.build_folder} && cmake -DCMAKE_BUILD_TYPE={build_type} .. && make"


        # Program path
        self.program_path = self.build_path + "MTS2D"
         
        # Generate conf file path and name
        self.conf_file = self.configObj.write_to_file(self.build_path)
        # Generate command to run simulation
        self.simulation_command = f"{self.program_path} {self.conf_file} {self.outputPath}"
        if self.useProfiling:
            self.simulation_command = "valgrind --tool=callgrind " + self.simulation_command


    def runSimulation(self, build=True):
        if build:
            self._build()
        # Start the timer right before running the command
        start_time = time.time()
        print("Running simulation")
        run_command(self.simulation_command)

        # Stop the timer right after the command completes
        end_time = time.time()

        # Calculate the duration
        duration = end_time - start_time
        return duration


    def _build(self):
        print("Building...")
        error = run_command(self.build_command)
        if error != 0:
            raise(Exception("Build error."))
        else:
            print("Build completed successfully.")
    

    def plot(self):
        #TODO this should just import the relevant functions
        plot_script = os.path.join(self.project_path, "SimulationScripts/Plotting/plotAll.py")
        plot_command = f"python3 {plot_script} {self.conf_file} {self.outputPath}"
        run_command(plot_command)
        

# The reason why this is so complicated is that if we simply use .readline(), it
# will not flush properly for lines that should be overwritten using \r.
def run_command(command, echo=True):
    if echo:
        # Simply print the command without colors or formatting
        print("Executing command:", command)

    # Start the process
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Buffer to store the output until we hit a line ending
    output_buffer = bytearray()

    while True:
        # Read one byte at a time
        byte = process.stdout.read(1)
        if byte:
            # Append the byte to the buffer
            output_buffer += byte

            # If the byte is a line ending, decode and print the buffer
            if byte in (b'\n', b'\r'):
                # Decode the buffer and print it
                print(output_buffer.decode('utf-8', errors='replace'), end='')
                # Clear the buffer
                output_buffer.clear()
        else:
            if process.poll() is not None:
                break

    # Output any remaining bytes in the buffer after the process has ended
    if output_buffer:
        print(output_buffer.decode('utf-8', errors='replace'), end='')

    # Check for any errors
    err = process.stderr.read().decode('utf-8')
    if err:
        print("Error:", err)

    return process.returncode

def findOutputPath(logging=True, createOutputFolder=True, outputFolderName="2DCS_output"):
    # Define the paths to check
    paths = ["/media/elias/dataStorage/", "/data2/elundheim/", "/data/elundheim/"]

    # Initialize a variable to store the chosen path
    chosen_path = None

    # Iterate through the paths and check if they exist
    for path in paths:
        if os.path.exists(path):
            chosen_path = path
            break  # Stop the loop once a valid path is found

    # Check if a valid path was found or raise an error
    if chosen_path is None:
        raise FileNotFoundError("None of the provided paths exist.")
    
    # Create the output folder if it does not exist
    if createOutputFolder:
        full_output_path = os.path.join(chosen_path, outputFolderName)+'/'
        if not os.path.exists(full_output_path):
            os.makedirs(full_output_path)
    else:
        full_output_path = chosen_path

    if(logging):
        print(f"Chosen output path: {full_output_path}")
    return full_output_path

if __name__ == "__main__":
    print(findOutputPath(logging=False))