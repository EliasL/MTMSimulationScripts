from pathlib import Path
import subprocess
import time
import shutil
import os
import sys
import platform
import glob


# Add Management to sys.path (used to import files)
sys.path.append(str(Path(__file__).resolve().parent.parent / "Plotting"))
# Now we can import from Management
from settings import settings


class SimulationManager:
    def __init__(
        self,
        configObj,
        outputPath=None,
        debugBuild=False,
        useProfiling=False,
        overwriteData=False,
        taskName=None,
    ):
        self.configObj = configObj
        self.taskName = taskName
        self.outputPath = findOutputPath() if outputPath is None else outputPath

        self.useProfiling = useProfiling
        self.parent_folder = str(Path(__file__).resolve().parent.parent.parent)
        self.project_path = os.path.join(self.parent_folder, "MTS2D")
        self.script_path = os.path.join(self.parent_folder, "SimulationScripts")
        # Change the working directory
        os.chdir(self.project_path)

        # Build folder
        self.debugBuild = debugBuild
        self.release_build_folder = "build-release/"
        self.profile_build_folder = "build/"
        self.build_folder = (
            self.profile_build_folder if debugBuild else self.release_build_folder
        )
        run_command(f"mkdir -p {self.build_folder}")
        # Build path
        self.build_path = os.path.join(self.project_path, self.build_folder)

        # I think it is better to always use release
        build_type = "Debug" if self.useProfiling else "Release"
        self.build_command = f"cd {self.build_folder} && cmake -DCMAKE_BUILD_TYPE={build_type} .. && make"

        # Program path
        self.program_path = self.build_path + "MTS2D"

        # Generate conf file path and name
        self.conf_file = self.configObj.write_to_file(self.build_path)
        self.subfolderName = Path(self.conf_file).stem
        # Generate command to run simulation
        self.simulation_command = f"{self.program_path} -c {self.conf_file} -o {self.outputPath} {' -r' if overwriteData else ''}"
        if self.useProfiling and platform.system() == "Linux":
            self.simulation_command = f"LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/7/libasan.so valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes {self.simulation_command}"

    def runSimulation(self, build=True, resumeIfPossible=True, silent=False):
        if build:
            self.build()

        if resumeIfPossible:
            dump = None
            try:
                dump = self.findDumpFile(0)
            except Exception as e:
                print(e)
                pass
            if dump is not None:
                # We resume instead of starting normally
                return self.resumeSimulation(silent=silent, build=build)

        # Start the timer right before running the command
        start_time = time.time()
        print("Running simulation")
        run_command(self.simulation_command, echo=not silent, taskName=self.taskName)

        # Stop the timer right after the command completes
        end_time = time.time()
        # Calculate the duration
        duration = end_time - start_time

        return duration

    def resumeSimulation(
        self,
        index=0,
        name=None,
        dumpFile=None,
        build=True,
        overwriteSettings=False,
        overwriteData=False,
        silent=False,
        newOutput=False,
    ):
        if build:
            self.build()

        # if the name is set, we search for that file name,
        # otherwise, we sort the files by date created and choose the newest
        # (index 0)
        if dumpFile is None:
            dumpFile = self.findDumpFile(index, name)
        else:
            if " " in dumpFile:
                raise ValueError(f"Dump path cannot contain white space! {dumpFile}")

        start_time = time.time()
        # We can choose to use the previous settings, or overwrite them using new ones
        # Initialize the base command
        command = [self.program_path, "-d", dumpFile]

        # Conditionally add flags and paths based on inputs
        if overwriteSettings:
            command.extend(["-c", self.conf_file])

        if newOutput:
            # Output does not specify the folder, but only the storage drive path
            # The data folder inside the output folder is completely determined by
            # the name variable in the config file
            command.extend(["-o", self.outputPath])

        if overwriteData:
            # If the data folder already contains a csv macrodata file with
            # a load value equal to the maxLoad value, it will not run and
            # overwrite values unless this flag is set
            command.append("-r")

        # Join the command list into a single string
        final_command = " ".join(command)

        # Now pass the final command to the run_command function
        run_command(
            final_command,
            echo=not silent,
            taskName=self.taskName,
        )
        # Stop the timer right after the command completes
        end_time = time.time()
        # Calculate the duration
        duration = end_time - start_time

        return duration

    def findDumpFile(self, index=0, name=None):
        """
        Find the dump file in the specified folder by name or by index after sorting by creation date.

        :param index: Index of the file to retrieve after sorting by creation date (default newest).
        :param name: Name of the dump file to find. If specified, index is ignored.
        :return: Path to the dump file.
        """

        dumpFolderPath = os.path.join(
            self.outputPath, self.subfolderName, settings["DUMPFOLDERPATH"]
        )

        # Check if a specific file name is given
        if name:
            # Search for the file with the given name
            for file in glob.glob(os.path.join(dumpFolderPath, "*")):
                if name in os.path.basename(file):
                    return file
            raise FileNotFoundError(f"No file named {name} found in {dumpFolderPath}")

        # If no specific file name is given, sort files by creation time
        files = list(
            filter(os.path.isfile, glob.iglob(os.path.join(dumpFolderPath, "*")))
        )
        files.sort(
            key=os.path.getmtime, reverse=True
        )  # Sort files by modification time, newest first

        # Return the file at the specified index
        try:
            return files[index]
        except IndexError:
            if len(files) == 0:
                raise Warning("No dumps found.")
            else:
                raise IndexError(
                    f"No file at index {index}. Only {len(files)} files available in {dumpFolderPath}."
                )

    def clean(self):
        # Print a message to indicate the cleaning process has started
        print("Cleaning...")

        # Construct the full path to the build directory
        build_dir_path = os.path.join(self.project_path, self.build_folder)

        # Check if the build directory exists
        if os.path.exists(build_dir_path):
            # Use shutil.rmtree to remove the directory and all its contents
            shutil.rmtree(build_dir_path)
            print(f"Removed build directory: {build_dir_path}")
        else:
            print(f"Build directory does not exist: {build_dir_path}")

        # Optionally, recreate the build directory to maintain structure
        os.makedirs(build_dir_path)
        print(f"Recreated build directory: {build_dir_path}")

    def build(self, autoClean=False):
        moduleCommand = self.loadModulesCommand()
        if moduleCommand:
            self.build_command = moduleCommand + self.build_command
        print("Building...")
        error = run_command(self.build_command, taskName=self.taskName)
        if error != 0:
            if autoClean:
                Warning("Build failed! Attempting to clean and rebuild")
                self.clean()
                error = run_command(self.build_command, taskName=self.taskName)
                if error != 0:
                    raise (Exception(f"Build error! Error code {error}"))
            else:
                raise (Exception(f"Build error! Error code {error}"))
        else:
            print("Build completed successfully.")

    def plot(self):
        # We import this inside the function so that we can choose not to import
        # if we don't want to plot
        # Add Management to sys.path (used to import files)
        sys.path.append(str(Path(__file__).resolve().parent.parent / "Plotting"))
        from plotAll import plotAll

        plotAll(self.conf_file, self.outputPath)
        pass

    def loadModulesCommand(self):
        # If cmake is not reccognized
        if run_command("which cmake") == 0:
            # No need to load modules
            return None
        else:
            # Then we load the modules
            command = "module load cmake llvm/15.0.6 && "
            print("Loading modules: cmake")
            return command


# The reason why this is so complicated is that if we simply use .readline(), it
# will not flush properly for lines that should be overwritten using \r.
def run_command(command, echo=True, taskName=None):
    if taskName is None:
        taskName = ""
    elif not (taskName[0] == "[" and taskName[-1] == "]"):
        taskName = f"[{taskName}]"

    if echo:
        # Simply print the command without colors or formatting
        print(f"{taskName} Executing command:", command)

    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # executable="/bin/bash",  # Use bash instead of default /bin/sh
        env=os.environ,
    )

    # Buffer to store the output until we hit a line ending
    output_buffer = bytearray()

    while True:
        # Read one byte at a time
        byte = process.stdout.read(1)
        if byte:
            # Append the byte to the buffer
            output_buffer += byte

            # If the byte is a line ending, decode and print the buffer
            if byte in (b"\n", b"\r"):
                # Decode the buffer and print it with self.name as a prefix
                print(
                    f"{taskName} {output_buffer.decode('utf-8', errors='replace')}",
                    end="",
                )
                # Clear the buffer
                output_buffer.clear()
        else:
            if process.poll() is not None:
                break

    # Output any remaining bytes in the buffer after the process has ended
    if output_buffer:
        print(f"{taskName} {output_buffer.decode('utf-8', errors='replace')}", end="")

    # Check for any errors
    err = process.stderr.read().decode("utf-8")
    if err:
        print(f"{taskName} Error:", err)

    return process.returncode


def findOutputPath(
    logging=True, createOutputFolder=True, outputFolderName="MTS2D_output"
):
    # Define the paths to check
    paths = [
        "/Volumes/data/",
        "/media/elias/dataStorage/",
        "/data2/elundheim/",
        "/data/elundheim/",
        "/lustre/fswork/projects/rech/bph/uog82gz/",  # JeanZay
        "/Users/elias/Work/PhD/Code/localData/",
        "/tmp/",
    ]

    # Initialize a variable to store the chosen path
    chosen_path = None

    # Iterate through the paths and check if they exist
    for path in paths:
        if os.path.exists(path):
            chosen_path = path
            break  # Stop the loop once a valid path is found

    if chosen_path == "/tmp/":
        print("Warning: Using temp output folder!")

    # Check if a valid path was found or raise an error
    if chosen_path is None:
        raise FileNotFoundError("None of the provided paths exist.")

    # Create the output folder if it does not exist
    if createOutputFolder:
        full_output_path = os.path.join(chosen_path, outputFolderName) + "/"
        if not os.path.exists(full_output_path):
            os.makedirs(full_output_path)
    else:
        full_output_path = chosen_path

    if logging:
        print(f"Chosen output path: {full_output_path}")
    return full_output_path


if __name__ == "__main__":
    print(findOutputPath(logging=False))
