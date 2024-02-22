import os

class SimulationConfig:

    """
    When adding or removing config settings, remember to also update
    paramParser.cpp and simulation.cpp.
    """

    def __init__(self, **kwargs):
        # Simulation Settings
        self.nx = 10  # Default = 10
        self.ny = 10  # Default = 10
        self.nrThreads = 1  # Default = 1
        self.seed = 0 # Default = 0
        self.plasticityEventThreshold = 0.2 # Default 0.2

        # Loading parameters
        self.startLoad = 0.0  # Default = 0.0
        self.loadIncrement = 0.01  # Default = 0.01
        self.maxLoad = 1.0  # Default = 1.0
        self.noise = 0.05 # Default = 0.05

        # Tolerances and Iterations
        self.nrCorrections = 10 # Default = 10
        self.epsg = 0.0  # Default = 0.0
        self.epsf = 0.0  # Default = 0.0
        self.epsx = 0.0  # Default = 0.0
        self.maxIterations = 0  # Default = 0 (Translates to unlimited iterations)

        # Logging settings
        self.showProgress = 1 # Default 1 (Can be either 0, 1 or 2. Nothing, minimal, and progress bar)

        # Update with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise(AttributeError(f"Unkown keyword: {key}"))

    def generate_name(self, withExtension=True):
        name = (
            f"s{self.nx}x{self.ny}"+
            f"l{self.startLoad},{self.loadIncrement},{self.maxLoad}"+
            f"t{self.nrThreads}"
        )
        # Conditionally append tolerances and iterations if they are not default
        if self.noise != 0.05:
            name += f"n{self.noise}"
        if self.nrCorrections != 10:
            name += f"m{self.nrCorrections}"
        if self.epsg != 0.0:
            name += f"EpsG{self.epsg}"
        if self.epsf != 0.0:
            name += f"EpsF{self.epsf}"
        if self.epsx != 0.0:
            name += f"EpsX{self.epsx}"
        if self.maxIterations != 0:
            name += f"MaxIter{self.maxIterations}"
        if self.plasticityEventThreshold != 0.2:
            name += f"PET{self.plasticityEventThreshold}"

        # We always add the seed at the very end
        name += f"s{self.seed}"

        if withExtension:
            # Add file extension
            name += ".conf"

        return name
    
    def get_path_and_name(self, path, withExtension=True):
        filename = self.generate_name(withExtension)
        full_path = os.path.join(path, filename)  # Corrected line
        return full_path

    def write_to_file(self, path):
        full_path=self.get_path_and_name(path)
        
        with open(full_path, 'w') as file:
            file.write("# Simulation Settings\n")
            for attr, value in self.__dict__.items():
                file.write(f"{attr} = {value} # Default = {value}\n")

        return full_path

    def parse(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No config file found at {path}")
        
        with open(path, 'r') as file:
            for line in file:
                # Ignore comments
                if line.startswith('#') or line.strip() == '':
                    continue
                
                # Remove everything after comment
                line = line.split('#')[0]

                # Parse the attribute and its value
                parts = line.split('=')
                if len(parts) != 2:
                    continue  # Skip lines that do not match the expected format
                
                attr, value = parts[0].strip(), parts[1].split('#')[0].strip()
                # Convert value to the correct type based on the attribute
                if hasattr(self, attr):
                    current_value = getattr(self, attr)
                    if isinstance(current_value, int):
                        value = int(value)
                    elif isinstance(current_value, float):
                        value = float(value)
                    # Assuming other types are strings, no conversion needed
                    
                    setattr(self, attr, value)

class ConfigGenerator:
    @staticmethod
    def generate_over_(argument_name, values, **kwargs):
        """
        Generate a list of SimulationConfig objects over a user-selected argument.

        :param argument_name: The name of the argument to vary (e.g., 'nrThreads', 'seed').
        :param values: A list of values for the specified argument.
        :param kwargs: Additional keyword arguments to pass to each SimulationConfig object.
        :return: A list of SimulationConfig objects with varying values for the specified argument.
        """
        configs = []
        for value in values:
            # Use **kwargs to pass other fixed arguments, and update the varying argument dynamically.
            config_kwargs = kwargs.copy()
            config_kwargs[argument_name] = value
            configs.append(SimulationConfig(**config_kwargs))
        return configs

    @staticmethod
    def generate_over_threads(threads_list, **kwargs):
        return [SimulationConfig(nrThreads=threads, **kwargs) for threads in threads_list]

    @staticmethod
    def generate_over_seeds(seeds, **kwargs):
        return [SimulationConfig(seed=seed, **kwargs) for seed in seeds]
    
if __name__ == "__main__":
    import os

    conf = SimulationConfig()
    conf.loadIncrement=0.005
    conf.maxLoad=1
    path = conf.write_to_file('build/')
    # Extract the directory part from the original path
    directory = os.path.dirname(path)
    
    # Define the new file name (keep the same directory)
    new_file_name = 'smallSimulation.conf'  # Replace 'new_filename.ext' with the new name
    
    # Construct the new path with the same directory but a new file name
    new_path = os.path.join(directory, new_file_name)
    
    # Rename the file
    os.rename(path, new_path)

    