from itertools import product
import os
from collections import OrderedDict


class SimulationConfig:
    """
    When adding or removing config settings, remember to also update
    paramParser.cpp and simulation.cpp.
    """

    def __init__(self, configPath=None, **kwargs):
        # Simulation Settings
        self.rows = 3
        self.cols = 3
        self.usingPBC = 1  # 0=False, 1=True
        self.scenario = "simpleShear"
        self.nrThreads = 1
        self.seed = 0
        self.QDSD = 0.00  # Quenched dissorder standard deviation
        self.initialGuessNoise = 0.05

        # Loading parameters
        self.startLoad = 0.0
        self.loadIncrement = 1e-5
        self.maxLoad = 1.0

        # Minimizer settings
        self.minimizer = "LBFGS"  # FIRE / LBFGS / CG
        # - LBFGS
        self.LBFGSNrCorrections = 10
        self.LBFGSScale = 1.0
        self.LBFGSEpsg = 0.0
        self.LBFGSEpsf = 0.0
        self.LBFGSEpsx = 0.0
        self.LBFGSMaxIterations = 0
        # - Conjugate Gradient
        self.CGScale = 1.0
        self.CGEpsg = 0.0
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
        self.dtMin = self.dtStart * 0.000001
        self.maxCompS = 0.01
        self.eps = 0.000
        self.epsRel = 0.0
        self.delta = 0.0
        self.maxIt = 10000

        # Logging settings
        self.plasticityEventThreshold = 0.1
        self.showProgress = 1

        if configPath is not None:
            self.parse(configPath)

        # Update with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                # Check if the given value is the correct type
                current_value = getattr(self, key)
                if not isinstance(value, type(current_value)):
                    raise TypeError(
                        f"{key} should be given a {type(current_value)} but was given a {type(value)}."
                    )
                setattr(self, key, value)
            elif key != "NONAME":
                raise (Warning(f"Unkown keyword: {key}"))

        if "NONAME" not in kwargs.keys():
            self.name = self.generate_name(withExtension=False)

    def generate_name(self, withExtension=True):
        name = (
            self.scenario + ","
            f"s{self.rows}x{self.cols}"
            + f"l{self.startLoad},{self.loadIncrement},{self.maxLoad}"
            + f"{'PBC' if self.usingPBC == 1 else 'NPBC'}"
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
                if attr != "NONAME":
                    file.write(f"{attr} = {value}\n")

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
        Generate a list of SimulationConfig objects with combinations prioritizing the 'seed' parameter.

        This method first ensures that combinations involving the 'seed' parameter are generated such that
        the 'seed' values are the first to iterate over. It helps in generating configurations where the 'seed'
        changes prior to other parameters, useful for setups where seed initialization is critical.

        Parameters:
            kwargs: A dictionary of argument names to iterables of possible values, where 'seed' is treated
                    as a special parameter to be prioritized in combinations.

        Returns:
            A tuple containing a list of SimulationConfig objects and a list of labels describing the configurations
            with only varying parameters.
        """
        # Separate 'seed' from other parameters if it's present
        seeds = kwargs.pop(
            "seed", [None]
        )  # Default to [None] if 'seed' is not provided
        kwargs = sorted(kwargs.items())

        # Prepare the remaining arguments, ensuring they are iterable
        processed_kwargs = OrderedDict()
        for key, value in kwargs:
            if isinstance(value, str) or not isinstance(value, list):
                processed_kwargs[key] = [value]  # Treat single values as a list
            else:
                processed_kwargs[key] = value

        # Generate all combinations of non-seed parameters
        non_seed_combinations = list(product(*processed_kwargs.values()))

        # Check if seeds is a single int
        seeds = [seeds] if isinstance(seeds, int) else seeds
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
            num_seeds = len(seeds)
            nr_groups = (
                len(configs) // num_seeds
            )  # Ensure integer division for grouping

            # Iterate over the number of seed groups, not over seeds or configs directly
            for index in range(nr_groups):
                # Calculate the slice indices for both configs and labels
                start = index * num_seeds
                end = start + num_seeds

                # Slice configs and labels according to calculated indices
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

    config = SimulationConfig(loadIncrement=0.01, minimizer="CG", nrThreads=1)
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
