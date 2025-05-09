from itertools import product
import os
from collections import OrderedDict
from collections.abc import Iterable
import subprocess
from pathlib import Path


class SimulationConfig:
    """
    When adding or removing config settings, remember to also update
    paramParser.cpp and simulation.cpp.
    """

    def __init__(self, configPath=None, **kwargs):
        # Simulation Settings
        self.rows = 3
        self.cols = 3
        self.usingPBC = "true"
        self.scenario = "simpleShear"
        self.nrThreads = 1  # This needs to be 1. Don't change. (see queueLocalJobs)
        self.seed = 0
        self.QDSD = 0.00  # Quenched dissorder standard deviation
        self.initialGuessNoise = 0.05
        self.meshDiagonal = "major"

        # Loading parameters
        self.startLoad = 0.0
        self.loadIncrement = 1e-5
        self.maxLoad = 1.0

        # Minimizer settings
        self.minimizer = "LBFGS"  # FIRE / LBFGS / CG
        self.epsR = 1e-20  # stopping criteria - Residual foce
        # - LBFGS
        self.LBFGSNrCorrections = 3  # nr correction vector paris, variable m in A Limited Memory Algorithm for Bound Constrained Optimization
        self.LBFGSScale = 1.0
        self.LBFGSEpsg = 1e-15
        self.LBFGSEpsf = 0.0
        self.LBFGSEpsx = 0.0
        self.LBFGSMaxIterations = 0
        # - Conjugate Gradient
        self.CGScale = 1.0
        self.CGEpsg = 1e-15
        self.CGEpsf = 0.0
        self.CGEpsx = 0.0
        self.CGMaxIterations = 0
        # - FIRE
        self.finc = 1.1
        self.fdec = 0.5
        self.alphaStart = 0.1
        self.falpha = 0.99
        self.dtStart = 0.01
        self.dtMax = self.dtStart * 3
        self.dtMin = self.dtStart * 1e-10
        self.maxCompS = 0.01
        self.eps = 1e-15
        self.epsRel = 0.0
        self.delta = 0.0
        self.maxIt = 200000

        # Logging settings
        # Saves the mesh if number of plastic events (npe) > number of
        # elements(ne) * plasticityEventThreshold (t).
        # if npe > ne*t:
        #   save frame
        self.logDuringMinimization = 0  # 0=False, 1=True
        self.plasticityEventThreshold = 0.05
        self.energyDropThreshold = 1e-4
        self.showProgress = 1  # 0=False, 1=True

        if configPath is not None:
            self.parse(configPath)

        # Update with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                if value is not None:
                    # Check if the given value is the correct type
                    current_value = getattr(self, key)
                    if not isinstance(value, type(current_value)):
                        raise TypeError(
                            f"{key} should be given a {type(current_value)} but was given a {type(value)}."
                        )
                    setattr(self, key, value)
                else:
                    # If the value is none, we ignore it
                    continue
            elif key == "L":
                self.rows = value
                self.cols = value
            elif key != "NONAME":
                raise (Warning(f"Unkown keyword: {key}"))

        assert self.usingPBC.lower() == "true" or self.usingPBC.lower() == "false", (
            "Only use true or false"
        )

        if "NONAME" not in kwargs.keys():
            self.name = self.generate_name(withExtension=False)

    def generate_name(self, withExtension=True):
        name = (
            self.scenario + ","
            f"s{self.rows}x{self.cols}"
            + f"l{self.startLoad},{self.loadIncrement},{self.maxLoad}"
            + f"{'PBC' if self.usingPBC.lower() == 'true' else 'NPBC'}"
            + f"t{self.nrThreads}"
        )

        defaultValues = vars(SimulationConfig(NONAME=True))
        for attr, value in vars(self).items():
            if attr not in [
                "name",
                "scenario",
                "rows",
                "cols",
                "startLoad",
                "loadIncrement",
                "maxLoad",
                "usingPBC",
                "nrThreads",
                "seed",
            ]:
                if defaultValues.get(attr) != value:
                    name += f"{attr}{value}"

        # Seed should be appended once, and only once, at the very end
        name += f"s{self.seed}"

        if withExtension:
            name += ".conf"

        if "_" in name:
            raise AttributeError("The name is not allowed to contain '_'!")
        self.validate_threshold()

        return name

    def validate_threshold(self):
        if self.minimizer == "LBFGS":
            if self.LBFGSEpsf == 0 and self.LBFGSEpsg == 0 and self.LBFGSEpsx == 0:
                raise AttributeError("No threshold set!")
        elif self.minimizer == "CG":
            if self.CGEpsf == 0 and self.CGEpsg == 0 and self.CGEpsx == 0:
                raise AttributeError("No threshold set!")
        elif self.minimizer == "FIRE":
            if self.eps == 0 and self.epsRel == 0:
                raise AttributeError("No threshold set!")

    def get_path_and_name(self, path, withExtension=True):
        filename = self.generate_name(withExtension)
        full_path = os.path.join(path, filename)  # Corrected line
        return full_path

    @staticmethod
    def get_git_tag_of_MTS2D():
        # Determine the project path (three levels up + "MTS2D")
        parent_folder = str(Path(__file__).resolve().parent.parent.parent)
        project_path = os.path.join(parent_folder, "MTS2D")

        # Get the current git tag, executing the command in the specified directory
        try:
            git_tag = (
                subprocess.check_output(
                    ["git", "describe", "--tags", "--always"],
                    stderr=subprocess.DEVNULL,
                    cwd=project_path,  # Specify the project directory
                )
                .strip()
                .decode("utf-8")
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_tag = "unknown"  # Fallback if no git tag is found
        return git_tag

    def write_to_file(self, path):
        # Get the full path for the config file
        full_path = self.get_path_and_name(path)
        git_tag = self.get_git_tag_of_MTS2D()

        with open(full_path, "w") as file:
            # Add the git tag as a commented line at the top
            file.write(f"# Version: {git_tag}\n")
            file.write("# Simulation Settings\n")

            # Write the attributes and their values
            for attr, value in self.__dict__.items():
                if attr != "NONAME":
                    file.write(f"{attr} = {value}\n")

        return full_path

    def parse(self, path):
        # Convert to a string, if necessary, using os.fspath
        path = os.fspath(path)
        # Remove leading/trailing whitespace
        path = path.strip()
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

                attr, value = parts[0].strip(), parts[1].strip()
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
    def generate(group_by_seeds=False, **kwargs):
        """
        This function creates all combinations of configurations. If a config
        object is given where the method, seeds and steps are lists like so:
        method=[LBFGS, CG]
        seeds=[1,2]
        step=[0.1,0.2],
        This function will generate 2x2x2=8 config objects.

        The structure of the objects will by default be such that the methods
        are separated in their own lists:

        [
            [(LBFGS, 1,0.1), (LBFGS, 1,0.2), (LBFGS, 2,0.1), (LBFGS, 2,0.2)],
            [(CG,    1,0.1), (CG,    1,0.2), (CG,    2,0.1), (CG,    2,0.2)],
        ]

        If group_by_seeds is True, the groups are instead decided by the seed

        [
            [(LBFGS, 1,0.1), (LBFGS, 1,0.2), (CG, 1,0.1), (CG, 1,0.2)], # seed 1
            [(LBFGS, 2,0.1), (LBFGS, 2,0.2), (CG, 2,0.1), (CG, 2,0.2)], # seed 2
        ]

        """

        # Separate 'seed' from other parameters if it's present
        seeds = kwargs.pop(
            "seed", [None]
        )  # Default to [None] if 'seed' is not provided

        # Ensure 'seeds' is a list
        if isinstance(seeds, int):
            seeds = [seeds]
        elif isinstance(seeds, str) or not isinstance(seeds, Iterable):
            seeds = [seeds]
        else:
            seeds = list(seeds)

        # Prepare the remaining arguments, ensuring they are iterable
        processed_kwargs = OrderedDict()
        for key, value in kwargs.items():
            if isinstance(value, str) or not isinstance(value, Iterable):
                processed_kwargs[key] = [value]  # Treat single values as a list
            else:
                processed_kwargs[key] = list(value)

        # Generate all combinations of non-seed parameters
        non_seed_combinations = list(product(*processed_kwargs.values()))

        # Generate full combinations, prioritizing seeds
        combined = [
            (seed,) + combo for combo in non_seed_combinations for seed in seeds
        ]

        # Create SimulationConfig instances and generate labels focusing on varying parameters
        configs = []
        labels = []
        for values in combined:
            seed = values[0]
            non_seed_params = dict(zip(processed_kwargs.keys(), values[1:]))
            params = {"seed": seed, **non_seed_params}
            config = SimulationConfig(**params)

            # Determine which parameters are varying to include in the label
            varying_params = {
                k: v for k, v in non_seed_params.items() if len(processed_kwargs[k]) > 1
            }
            if len(seeds) > 1:  # Include 'seed' in label only if it varies
                varying_params["seed"] = seed
            label = ", ".join(f"{k}={v}" for k, v in varying_params.items())

            configs.append(config)
            labels.append(label)

        if group_by_seeds:
            grouped_configs = []
            grouped_labels = []
            num_combos = len(non_seed_combinations)
            for i in range(num_combos):
                start = i * len(seeds)
                end = start + len(seeds)
                grouped_configs.append(configs[start:end])
                grouped_labels.append(labels[start:end])
            return grouped_configs, grouped_labels
        else:
            return configs, labels

    @staticmethod
    def get_kwargs(configs):
        """
        A reverse of the generate method
        """

        # Create a default instance to compare against
        default_config = SimulationConfig()

        # Initialize dictionary to hold parameter values
        param_values = OrderedDict()

        if isinstance(configs, SimulationConfig):
            configs = [configs]

        # Populate the dictionary with parameter values from each config
        for config in configs:
            # __dict__ should be ordered
            for key, value in sorted(config.__dict__.items()):
                if key == "name":
                    continue
                if key not in param_values:
                    param_values[key] = set()
                param_values[key].add(value)

        # Convert sets to lists, remove parameters that match the default when there's only one value
        kwargs = OrderedDict()
        for key, values in param_values.items():
            # Convert set to list for easier manipulation
            value_list = list(values)

            if len(value_list) == 1:
                # If only one value and it is different from the default, save it directly
                if value_list[0] != getattr(default_config, key, None) or key == "seed":
                    kwargs[key] = value_list[0]
            else:
                # If more than one value, save it as a list
                kwargs[key] = value_list

        return kwargs

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

    @staticmethod
    def splitKwargs(kwargs):
        """
        Splits the given kwargs dictionary into two dictionaries:
        one containing keys corresponding to properties of the SimulationConfig class,
        and one containing the rest.

        Parameters:
            kwargs (dict): The input dictionary to split.

        Returns:
            tuple: A tuple of two dictionaries:
                - The first dictionary contains keys matching properties of SimulationConfig.
                - The second dictionary contains all other keys.
        """

        # Retrieve properties of the SimulationConfig class
        config_properties = set(dir(SimulationConfig()))

        # Dictionaries for matching and non-matching keys
        matching_keys = {}
        non_matching_keys = {}

        # Iterate over kwargs to classify keys
        for key, value in kwargs.items():
            if key in config_properties:
                matching_keys[key] = value
            else:
                non_matching_keys[key] = value

        return matching_keys, non_matching_keys


def get_custom_configs(scenario="large"):
    if scenario == "large":
        return SimulationConfig(
            rows=100,
            cols=100,
            startLoad=0.15,
            nrThreads=5,
            loadIncrement=1e-5,
            maxLoad=1.0,
        )

    elif scenario == "periodicBoundaryTest":
        return SimulationConfig(
            rows=4,
            cols=4,
            startLoad=0.0,
            nrThreads=4,
            loadIncrement=0.0001,
            maxLoad=1,
            scenario="periodicBoundaryTest",
        )

    elif scenario == "singleDislocation":
        return SimulationConfig(
            rows=6,
            cols=6,
            startLoad=0.0,
            nrThreads=6,
            loadIncrement=1e-5,
            maxLoad=0.001,
            scenario="singleDislocation",
        )

    elif scenario == "longSim":
        return SimulationConfig(
            rows=60,
            cols=60,
            startLoad=0.15,
            nrThreads=4,
            loadIncrement=1e-5,
            maxLoad=10.0,
            # scenario="simpleShearPeriodicBoundary")
            scenario="cyclicSimpleShear",
        )


if __name__ == "__main__":
    import os
    import sys

    L = 3
    config = SimulationConfig(
        usingPBC="true",
        scenario="simpleShear",
        rows=L,
        cols=L,
        startLoad=0.15,
        loadIncrement=0.001,
        minimizer="LBFGS",
        nrThreads=3,
        epsR=1e-5,  # LBFGSEpsg=1e-5
    )
    # config = get_custom_configs()
    if len(sys.argv) >= 2:
        scenario = sys.argv[1]
        config.scenario = scenario
        # if there is a complete custom scenario, we replace the config
        if get_custom_configs(scenario) is not None:
            config = get_custom_configs(scenario)
    build_path = os.path.join(os.getcwd(), "build/")
    path = config.write_to_file(build_path)
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
