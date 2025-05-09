import itertools
import os


class SimulationConfig:
    """
    When adding or removing config settings, remember to also update
    paramParser.cpp and simulation.cpp.
    """

    def __init__(self, **kwargs):
        # Simulation Settings

        self.rows = 10
        self.cols = 10
        self.usingPBC = 1  # 0=False, 1=True
        self.scenario = "simpleShear"
        self.nrThreads = 1
        self.seed = 0
        self.plasticityEventThreshold = 0.1

        # Loading parameters
        self.startLoad = 0.0
        self.loadIncrement = 0.00001
        self.maxLoad = 1.0
        self.noise = 0.05

        # Minimizer settings
        self.minimizer = "FIRE"  # FIRE / LBFGS
        self.nrCorrections = 10
        self.scale = 1.0
        self.epsg = 0.0
        self.epsf = 0.0
        self.epsx = 1.0e-6
        self.maxIterations = 0

        # Logging settings
        self.showProgress = 1

        # Update with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise (AttributeError(f"Unkown keyword: {key}"))
        self.validate()

    def validate(self, name=None):
        # we don't allow any '_' characters in the name
        if name is None:
            name = self.generate_name(True)

        count = name.count("_")
        if count != 0:
            raise (AttributeError("The name is not allowed to contain '_'!"))
        return True

    def generate_name(self, withExtension=True):
        name = (
            self.scenario + ","
            f"s{self.rows}x{self.cols}"
            + f"l{self.startLoad},{self.loadIncrement},{self.maxLoad}"
            + f"{'PBC' if self.usingPBC == 1 else 'NPBC'}"
            + f"t{self.nrThreads}"
        )
        # Conditionally append tolerances and iterations if they are not default
        if self.minimizer != "FIRE":
            name += self.minimizer
        if self.noise != 0.05:
            name += f"n{self.noise}"
        if self.nrCorrections != 10:
            name += f"m{self.nrCorrections}"
        if self.scale != 1:
            name += f"s{self.scale}"
        if self.epsg != 0.0:
            name += f"EpsG{self.epsg}"
        if self.epsf != 0.0:
            name += f"EpsF{self.epsf}"
        if self.epsx != 0.0:
            name += f"EpsX{self.epsx}"
        if self.maxIterations != 0:
            name += f"maxIter{self.maxIterations}"
        if self.plasticityEventThreshold != 0.1:
            name += f"PET{self.plasticityEventThreshold}"

        # We always add the seed at the very end
        name += f"s{self.seed}"

        if withExtension:
            # Add file extension
            name += ".conf"

        self.validate(name)
        return name

    def get_path_and_name(self, path, withExtension=True):
        filename = self.generate_name(withExtension)
        full_path = os.path.join(path, filename)  # Corrected line
        return full_path

    def write_to_file(self, path):
        full_path = self.get_path_and_name(path)

        with open(full_path, "w") as file:
            file.write("# Simulation Settings\n")
            for attr, value in self.__dict__.items():
                file.write(f"{attr} = {value} # Default = {value}\n")

        return full_path

    def parse(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No config file found at {path}")

        with open(path, "r") as file:
            for line in file:
                # Ignore comments
                if line.startswith("#") or line.strip() == "":
                    continue

                # Remove everything after comment
                line = line.split("#")[0]

                # Parse the attribute and its value
                parts = line.split("=")
                if len(parts) != 2:
                    continue  # Skip lines that do not match the expected format

                attr, value = parts[0].strip(), parts[1].split("#")[0].strip()
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
    def generate(**kwargs):
        """
        Generate a list of SimulationConfig objects over multiple arguments.

        Each keyword argument should be a pair where the key is the argument name
        and the value is an iterable of values for that argument.

        :param kwargs: Dictionary of argument names to iterables of possible values.
        :return: A list of SimulationConfig objects with all combinations of argument values.

        Example:
        configurations = ConfigGenerator.generate(nrThreads=[1, 2, 4], seed=[42, 43], scenario='simpleShear')
        for config in configurations:
            print(config.generate_name())
        """
        # Prepare the kwargs to ensure that each is a list (but keep strings intact)
        processed_kwargs = {}
        varying_keys = {}
        for key, value in kwargs.items():
            if isinstance(value, str) or not isinstance(value, list):
                # Handle strings and non-list non-string values as a single-item list
                processed_kwargs[key] = [value]
            else:
                # Use lists as is and mark varying keys
                processed_kwargs[key] = value
                if len(value) > 1:
                    varying_keys[key] = value

        # Extract argument names and corresponding lists of values
        keys = processed_kwargs.keys()
        values_product = itertools.product(*processed_kwargs.values())

        # Create a list of SimulationConfig objects and labels for each combination of argument values
        configs = []
        labels = []
        for values in values_product:
            params = dict(zip(keys, values))
            config = SimulationConfig(**params)
            # Create label containing only varying parameters
            label = ", ".join(f"{k}={params[k]}" for k in varying_keys if k in params)
            configs.append(config)
            labels.append(label)

        return configs, labels

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
        return [
            SimulationConfig(nrThreads=threads, **kwargs) for threads in threads_list
        ]

    @staticmethod
    def generate_over_seeds(seeds, **kwargs):
        return [SimulationConfig(seed=seed, **kwargs) for seed in seeds]


def get_custom_configs(scenario):
    if scenario == "periodicBoundaryTest":
        return SimulationConfig(
            rows=4,
            cols=4,
            startLoad=0.0,
            nrThreads=4,
            loadIncrement=0.0001,
            maxLoad=1,
            scenario="periodicBoundaryTest",
        )

    if scenario == "singleDislocation":
        return SimulationConfig(
            rows=6,
            cols=6,
            startLoad=0.0,
            nrThreads=6,
            loadIncrement=0.00001,
            maxLoad=0.001,
            scenario="singleDislocation",
        )

    if scenario == "longSim":
        return SimulationConfig(
            rows=60,
            cols=60,
            startLoad=0.15,
            nrThreads=4,
            loadIncrement=0.00001,
            maxLoad=10,
            # scenario="simpleShearPeriodicBoundary")
            scenario="cyclicSimpleShear",
        )


if __name__ == "__main__":
    import os
    import sys

    config = SimulationConfig()
    config.loadIncrement = 0.00001
    config.maxLoad = 1
    config.rows = 3
    config.cols = 3
    if len(sys.argv) >= 2:
        scenario = sys.argv[1]
        config.scenario = scenario
        # if there is a complete custom scenario, we replace the config
        if get_custom_configs(scenario) is not None:
            config = get_custom_configs(scenario)

    path = config.write_to_file("build/")
    # Extract the directory part from the original path
    directory = os.path.dirname(path)

    # Define the new file name (keep the same directory)
    new_file_name = (
        "smallSimulation.conf"  # Replace 'new_filename.ext' with the new name
    )

    # Construct the new path with the same directory but a new file name
    new_path = os.path.join(directory, new_file_name)

    # Rename the file
    os.rename(path, new_path)
