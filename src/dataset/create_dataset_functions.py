from src.functions import *
from omegaconf import OmegaConf
import os
import wandb
import pickle
import numpy as np
from pyDOE import lhs
from scipy.integrate import solve_ivp
import time

class ODE_modelling():
    def __init__(self, config):
        """
        Initializes an instance of ODE_modelling.
        Args:
            config: The configuration object containing various parameters.

        Attributes:
            config (object): The configuration object.
            modelling_method (str): The modelling method.
            model_flag (str): The model flag.
            time (str): The time interval for the simulation.
            num_of_points (int): The number of points for the simulation.
            init_condition_bounds (str): The bounds for initial conditions.
            sampling (str): The sampling method.
            init_conditions_dir (str): The directory for initial conditions.
            dataset_dir (str): The directory for saving the dataset.
            torch (bool): The flag to use PyTorch for the model.
            seed (int or None): The seed for random number generation.
        Methods:
            append_element: Appends an element to each state in the given value set by iterating over a range of values.
            append_element_set: Appends an element to each state in the given value set by iterating over a range of values.
            check_ic_yaml: Modeling guide contains all the variables that can be used in the modeling.
            create_init_conditions_set3: Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.
            solve: Solve the differential equations for the synchronous machine model.
            solve_sm_model: Solves the synchronous machine model for multiple initial conditions.
            save_dataset: Create and save dataset for the model.
            load_dataset: Load the dataset.
        """
        self.config = config
        self.modelling_method = config.modelling_method
        self.model_flag = config.model.model_flag
        self.time = config.time
        self.num_of_points = config.num_of_points
        self.init_condition_bounds = config.model.init_condition_bounds
        self.sampling = config.model.sampling
        self.init_conditions_dir = config.dirs.init_conditions_dir
        self.dataset_dir = config.dirs.dataset_dir
        self.torch = config.model.torch
        self.seed = None if not hasattr(config.model, 'seed') else config.model.seed

    
    def append_element(self, value_set, Value_range, num_ranges):
        """
        Appends an element to each state in the given value set by iterating over a range of values.

        Args:
            value_set (list): The list of states to which the element will be appended.
            Value_range (tuple): The range of values from which the element will be selected.
            num_ranges (int): The number of ranges to divide the Value_range into.

        Returns:
            list: A new list of states with the element appended.
        """
        new_value_set = []
        for j in range(len(value_set)):
            for i in range(num_ranges):
                value = (Value_range[0] + i * (Value_range[1] - Value_range[0]) / (num_ranges - 1) if num_ranges > 1 else Value_range[0])
                new_state = value_set[j].copy()
                new_state.extend([value])
                new_value_set.append(new_state)
        return new_value_set
    
    
    def append_element_set(self, value_set, Value_range, num_ranges):
        """
        Appends an element to each state in the given value set by iterating over a range of values.
        
        Args:
            value_set (list): The list of states to which the element will be appended.
            Value_range (tuple): The range of values from which the element will be selected.
            num_ranges (int): The number of ranges to divide the Value_range into.

        Returns:
            list: A new list of states with the element appended.
        """

        np.random.seed(self.seed if self.seed is not None else np.random.randint(0, 1000))
        if self.sampling=="Random":
            points = np.random.uniform(0, 1, num_ranges)
            points = points.reshape(-1, 1)
        elif self.sampling=="Linear":
            points = np.linspace(0, 1, num_ranges)
            points = points.reshape(-1, 1)
        elif self.sampling=="Lhs":
            points = lhs(n=1, samples=num_ranges)
        else:
            raise Exception("Sampling method not implemented")
        
        new_value_set = []
        iterations = len(value_set) if len(value_set)>1 else 1
        for j in range(iterations):
            if len(value_set)<1:
                values = (Value_range[0] + points * (Value_range[1] - Value_range[0]) if num_ranges > 1 else Value_range[0])
                new_value_set = [values][0].tolist()
            else:
                for i in points:
                    if isinstance(i, np.ndarray):
                        i = i.item()
                    value = (Value_range[0] + i * (Value_range[1] - Value_range[0]) if num_ranges > 1 else Value_range[0])
                    new_state = value_set[j].copy()
                    if isinstance(new_state, np.ndarray):
                        new_state = new_state.tolist()
                    new_state.extend([value])
                    new_value_set.append(new_state)
        return new_value_set

    def check_ic_yaml(self,init_conditions):
        """
        Checks if the variables in the initial conditions are present in the modeling guide and in the correct order.
        Also verifies that the iterations for each variable are integers.

        Args:
            init_conditions (list): List of initial conditions for which the model needs to be solved.

        Returns:
            None or ValueError: If the variables are not present in the modeling guide or the iterations are not integers.
        """
        modeling_guide_path = os.path.join(self.init_conditions_dir, "modellings_guide.yaml")
        modeling_guide = OmegaConf.load(modeling_guide_path)
        #check if proposed modeling is in the modeling guide
        for model in modeling_guide:
            model_name = model.get("name")
            if model_name == self.model_flag:
                keys = model.get("keys")
                self.fullname = model.get("fullname")
        if not keys:
            raise ValueError(f"Model {self.model_flag} not found in modeling guide")

        for i in range(len(init_conditions)):
            name = init_conditions[i].get("name")
            if name not in keys:
                raise ValueError(f"Variable {name} not found in modeling guide")
            if name != keys[i]: #same order as in the modeling guide
                raise ValueError(f"Variable {name} does not match the modeling variable {keys[i]}")
            #check if iterations are always a number:
            iterations = init_conditions[i].get("iterations")
            if not isinstance(iterations, int):
                raise ValueError(f"Variable {name} iterations must be an integer")
        return

    def create_init_conditions_set3(self):
        """
        Define the various initial conditions of the synchronous machine and return a matrix with all the possible combinations.

        Returns:
            list: A matrix with all the possible combinations of initial conditions.
        """
        if self.torch:# if using torch then use the nn_init_cond.yaml file to create collocation points init conditions
            init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"nn_init_cond"+str(self.init_condition_bounds)+".yaml")
        else: # 
            init_conditions_path = os.path.join(self.init_conditions_dir, self.model_flag,"init_cond"+str(self.init_condition_bounds)+".yaml")
        init_conditions = OmegaConf.load(init_conditions_path)
        self.check_ic_yaml(init_conditions)

        for i in range(len(init_conditions)):
            if len(init_conditions[i]["range"]) == 1: # if unique value then set iterations to 1
                init_conditions[i]["iterations"] = 1 

        # Initialize the set_of_values and iterations lists and variables
        set_of_values = []
        iterations = []
        variables = []
        # Extract values, iterations and variables from init_conditions
        for condition in init_conditions:
            set_of_values.append(condition['range'])
            iterations.append(condition['iterations'])
            variables.append(condition['name'])

        # Calculate the number of different initial conditions
        number_of_conditions = 1
        for it in iterations:
            number_of_conditions *= it

        if self.torch:
            print("Number of different initial conditions for collocation points: ", number_of_conditions)
            #wandb.log({"Number of different initial conditions for collocation points: ": number_of_conditions})
        else:
            print("Number of different initial conditions: ", number_of_conditions)
            #wandb.log({"Number of different initial conditions: ": number_of_conditions})

        print(variables, "Variables")
        print(set_of_values,"Set of values for init conditions")
        print(iterations,"Iterations per value")
        #wandb.log({"Set of values for init conditions: ": set_of_values})
        #wandb.log({"Iterations per value: ": iterations})
        init_condition_table = []   
        for k in range(len(set_of_values)):
            init_condition_table = self.append_element_set(init_condition_table, set_of_values[k], iterations[k])
        return init_condition_table

    def solve(self, x0, method, modelling_full):
        """
        Solve the differential equations for the synchronous machine model.

        Parameters:
        - x0: list of initial state variables

        Returns:
        - solution: solution of the differential equations
        """

        if method:
            solution = solve_ivp(modelling_full.odequations, self.t_span, x0, t_eval=self.t_eval)
        else:
            x0[1] = x0[1] / self.omega_B
            solution = solve_ivp(modelling_full.odequations_v2, self.t_span, x0, t_eval=self.t_eval)
        return solution

    def solve_sm_model(self, init_conditions, modelling_full, flag_time=False):
        """
        Solves the synchronous machine model for multiple initial conditions.

        Args:
            machine_params (dict): Dictionary containing the parameters of the synchronous machine.
            system_params (dict): Dictionary containing the parameters of the power system.
            init_conditions (list): List of initial conditions for which the model needs to be solved.

        Returns:
            list: List of solutions for each initial condition.

        """
        self.t_span, self.t_eval = set_time(self.time, self.num_of_points)
        solution_all=[]
        if flag_time:
            start_time = time.time()   
            start_per_iteration = start_time
            time_list = [] 
        for i in range(len(init_conditions)):
            solution = self.solve(init_conditions[i], self.modelling_method, modelling_full)
            solution_all.append(solution)
            if flag_time:
                end_per_iteration = time.time()
                time_list.append(end_per_iteration - start_per_iteration)
                start_per_iteration = end_per_iteration
            
        if flag_time:
            end_time = time.time()
            #print mean and std of time per iteration
            print("Mean time per iteration: ", np.mean(time_list), " and std: ", np.std(time_list))
            print(f"Time taken to solve the model for {len(init_conditions)} initial conditions: {end_time - start_time} seconds.")
        return solution_all
    
    def save_dataset(self, solution):
        """
        Create and save dataset for the model.

        Args:
            solution (list): The solution of the differential equations of the synchronous machine.

        Returns:
            list: The dataset of the synchronous machine.
        """
        dataset = []
        for i in range(len(solution)):
            r = [solution[i].t]  # append time to directory
            for j in range(len(solution[i].y)):
                r.append(solution[i].y[j])  # append the solution at each time step
            dataset.append(r)

        # check if folder exists if not create it
        if not os.path.exists(os.path.join(self.dataset_dir, self.model_flag)):
            os.makedirs(os.path.join(self.dataset_dir, self.model_flag))

        # count the number of files in the directory
        num_files = len([f for f in os.listdir(os.path.join(self.dataset_dir, self.model_flag)) if os.path.isfile(os.path.join(self.dataset_dir, self.model_flag, f))])
        print("Number of files in the directory: ", num_files)
        print(f'Saved dataset "{self.model_flag, "dataset_v" + str(num_files + 1)}".')
        #wandb.log({"Dataset saved": f'Saved dataset "{self.model_flag, "dataset_v" + str(num_files + 1)}".'})
        # save the dataset as pickle in the dataset directory
        dataset_path = os.path.join(self.dataset_dir, self.model_flag, "dataset_v" + str(num_files + 1) + ".pkl")
        with open(dataset_path, 'wb') as f:
            pickle.dump(dataset, f)

        return dataset

   

    def load_dataset(self, name):
        """
        Load the dataset.

        Args:
            name (str): The name of the dataset.

        Returns:
            list: The dataset of the synchronous machine.
        """
        dataset_path = os.path.join(self.dataset_dir, self.model_flag, name)
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

  



    