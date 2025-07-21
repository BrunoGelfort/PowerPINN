import numpy as np
from scipy.integrate import solve_ivp
from src.functions import set_time
import torch
import os
from omegaconf import OmegaConf

class SynchronousMachineModels():
    def __init__(self, config):
        """
        Initialize the synchronous machine model with the given configuration.

        Parameters:
            config (dict): The configuration of the model.
            
        Attributes:
            params_dir (str): The path to the parameters directory.
            machine_num (int): The number of the machine to be used.
            model_flag (str): The model to be used.
            define the parameters of the machine based on the machine_num
            define the parameters of the power system
        """

        self.params_dir = config.dirs.params_dir # path to the parameters directory
        self.machine_num = config.model.machine_num # the number of the machine to be used
        self.model_flag = config.model.model_flag # the model to be used
        self.define_machine_params() # define the parameters of the machine based on the machine_num
        self.define_system_params() # define the parameters of the power system
          

    def define_machine_params(self):
        """
        Define the parameters of the synchronous machine based on the machine_num
        and potentially the parameters of the AVR and the Governor.

        Returns:
            Attributes: The parameters of the synchronous machine the AVR and the Governor
        """
        machine_params_path = os.path.join(self.params_dir, "machine" + str(self.machine_num) + ".yaml") # path to the selected machine parameters
        machine_params = OmegaConf.load(machine_params_path)
        
        
        if not(self.model_flag=="SM_IB"):
            for param in ['X_d_dash', 'X_q_dash', 'H', 'D', 'T_d_dash', 'X_d', 'T_q_dash', 'X_q', 'E_fd', 'P_m']:
                setattr(self, param, getattr(machine_params, param))
        else:
            for param in ['H', 'D', 'X_d_dash', 'X_q_dash', 'P_m']:
                setattr(self, param, getattr(machine_params, param))

        if self.model_flag=="SM6":
            self.model_name = "6th order Synchronous Machine Model"
            for param in ['T_d_dash_dash', 'T_q_dash_dash', 'X_d_dash_dash', 'X_q_dash_dash']:
                setattr(self, param, getattr(machine_params, param))

        if self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV": 
            avr_params_path= os.path.join(self.params_dir,"avr.yaml") #path to the automatic voltage regulator parameters
            avr_params= OmegaConf.load(avr_params_path)
            for param in ['K_A', 'T_A', 'K_E', 'T_E', 'K_F', 'T_F', 'V_ref']:
                setattr(self, param, getattr(avr_params, param))
        else: 
            avr_params=None

        if self.model_flag=="SM_AVR_GOV":
            gov_params_path= os.path.join(self.params_dir,"gov.yaml") #path to the governor parameters
            gov_params= OmegaConf.load(gov_params_path)
            for param in ['P_c', 'R_d', 'T_ch', 'T_sv']:
                setattr(self, param, getattr(gov_params, param))
        else:
            gov_params=None

        return 
    
    def define_system_params(self):
        """
        Define the parameters of the power system.

        Returns:
            dict: The parameters of the power system.
        """
        system_params_path = os.path.join(self.params_dir, "system_bus.yaml")
        system_params = OmegaConf.load(system_params_path)
        for param in ['Vs', 'theta_vs', 'omega_B']:
            setattr(self, param, getattr(system_params, param))
        return 

    def calculate_currents(self, theta, E_d_dash, E_q_dash):
        """
        Calculates the currents I_d and I_q based on the given parameters.

        Parameters:
        theta (rad): The angle .
        E_d_dash (pu): The value of E_d_dash.
        E_q_dash (pu): The value of E_q_dash.

        Returns:
        tuple: A tuple containing the calculated values of I_d and I_q.
        """

        Rs=0.0
        Re=0.0
        Xep=0.1
        alpha = [[(Rs+Re), -(self.X_q_dash+Xep)], [(self.X_d_dash+Xep), (Rs+Re)]]
        
        inv_alpha = np.linalg.inv(alpha)
        # Calculate I_d and I_q
        if isinstance(theta, torch.Tensor):
            beta = [[E_d_dash - self.Vs*torch.sin(theta-self.theta_vs)], [E_q_dash - self.Vs*torch.cos(theta-self.theta_vs)]]
        else:
            beta = [[E_d_dash - self.Vs*np.sin(theta-self.theta_vs)], [E_q_dash - self.Vs*np.cos(theta-self.theta_vs)]]
            
        I_d= inv_alpha[0][0]*beta[0][0] + inv_alpha[0][1]*beta[1][0]
        I_q= inv_alpha[1][0]*beta[0][0] + inv_alpha[1][1]*beta[1][0]
        
        return I_d, I_q

    def calculate_voltages(self, theta, I_d, I_q):
        """
        Calculate the voltage V_t based on the given inputs, for AVR model

        Parameters:
        theta (rad): The angle in radians.
        I_d (pu): The d-axis current.
        I_q (pu): The q-axis current.

        Returns:
        float: The magnitude of the total voltage V_t(pu).
        """
        Re = 0.0
        Xep = 0.1
        if isinstance(theta, torch.Tensor):
            V_d = Re * I_d - Xep * I_q + self.Vs * torch.sin(theta - self.theta_vs)
            V_q = Re * I_q + Xep * I_d + self.Vs * torch.cos(theta - self.theta_vs)
            V_t = torch.sqrt(V_d ** 2 + V_q ** 2)
        else:
            V_d = Re * I_d - Xep * I_q + self.Vs * np.sin(theta - self.theta_vs)
            V_q = Re * I_q + Xep * I_d + self.Vs * np.cos(theta - self.theta_vs)
            V_t = np.sqrt(V_d ** 2 + V_q ** 2)  # equal to Vs
        return V_t
        
    def odequations(self, t, x):
        """
        Calculates the derivatives of the state variables for the synchronous machine model.

        Parameters:
            t (float): The current time.
            x (list): A list of state variables, different for each model type.

        Returns:
            list: A list of derivatives, different for each model type.
        """
        if self.model_flag=="SM_IB" or self.model_flag=="SM4":
            theta, omega, E_d_dash, E_q_dash = x
        if self.model_flag == "SM6":
            theta, omega, E_d_dash, E_q_dash, E_d_dash_dash, E_q_dash_dash = x
        if self.model_flag=="SM_AVR":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd = x
        if self.model_flag=="SM_AVR_GOV":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_m, P_sv = x
        if self.model_flag == "SM6_AVR":
            theta, omega, E_d_dash, E_q_dash, E_d_dash_dash, E_q_dash_dash, R_F, V_r, E_fd = x
        if self.model_flag == "SM6_AVR_GOV":
            theta, omega, E_d_dash, E_q_dash, E_d_dash_dash, E_q_dash_dash, R_F, V_r, E_fd, P_m, P_sv = x

        # Calculate currents from algebraic equations
        I_d, I_q = self.calculate_currents(theta, E_d_dash, E_q_dash)

        if (self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV" or self.model_flag== "SM6_AVR" or self.model_flag== "SM6_AVR_GOV"): # calculate V_t from algebraic equations
            V_t = self.calculate_voltages(theta, I_d, I_q)
        
        # Calculate theta derivative
        dtheta_dt = omega
        
        # Calculate omega derivative
        if self.model_flag=="SM_AVR_GOV" or  self.model_flag=="SM6_AVR_GOV": # calculate omega derivative from algebraic equations
            domega_dt = (self.omega_B / (2 * self.H)) * (P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega)
        else:
            domega_dt = (self.omega_B / (2 * self.H)) * (self.P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega)
        
        # Calculate E_dash derivatives
        if self.model_flag=="SM_IB":
            dE_q_dash_dt = 0
            dE_d_dash_dt = 0
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]
        
        if self.model_flag=="SM4": 
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + self.E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]
        
        if self.model_flag == "SM6":
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + self.E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            dE_q_dash_dash_dt = (1 / self.T_d_dash_dash) * (E_q_dash - E_q_dash_dash + I_d * (self.X_d_dash - self.X_d_dash_dash))
            dE_d_dash_dash_dt = (1 / self.T_q_dash_dash) * (E_d_dash - E_d_dash_dash - I_q * (self.X_q_dash - self.X_q_dash_dash))
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dE_d_dash_dash_dt, dE_q_dash_dash_dt]
        
        # Automatic Voltage Regulator (AVR) dynamics 4.46-4.48
        # Exciter and AVR equations

        if (self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV"):
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            dR_F_dt      = (1 / self.T_F) * (-R_F + (self.K_F / self.T_F) * E_fd)
            
            dV_r_dt     = (1 / self.T_A) * (-V_r + (self.K_A * R_F) - (self.K_A * self.K_F / self.T_F) * E_fd + self.K_A * (self.V_ref - V_t))
            """
            V_r_min = 0.8
            V_r_max = 9
            V_r = V_r.clone()
            dV_r_dt = dV_r_dt.clone()

            lower_limit__V_r_mask = (V_r <= V_r_min) & (dV_r_dt < 0)
            dV_r_dt[lower_limit__V_r_mask] = 0  # Apply condition only where mask is True
            V_r[lower_limit__V_r_mask] = V_r_min

            upper_limit_V_r_mask = (V_r >= V_r_max) & (dV_r_dt > 0)
            dV_r_dt[upper_limit_V_r_mask] = 0
            V_r[upper_limit_V_r_mask] = V_r_max
            """
            dE_fd_dt     = (1 / self.T_E) * (-(self.K_E + 0.098 * np.e**(E_fd*0.55)) * E_fd + V_r)
            
            if self.model_flag=="SM_AVR_GOV":        # Governor equations # recheck it after the meeting
                dP_m_dt  = (1 / self.T_ch) * (-P_m  + P_sv) # 4.110  from dynamics dT_m_dt = - P_m / (2 * H) + P_sv 4.100 + check draw.io
                dP_sv_dt = (1 / self.T_sv) * (-P_sv + self.P_c - (1/self.R_d) *(omega/self.omega_B))
                """
                P_sv_max = 1
                P_sv_min = 0
                P_sv = P_sv.clone()
                dP_sv_dt = dP_sv_dt.clone()

                lower_limit_P_sv_mask = (P_sv <= P_sv_min) & (dP_sv_dt < 0)
                dP_sv_dt[lower_limit_P_sv_mask] = 0
                P_sv[lower_limit_P_sv_mask] = P_sv_min

                upper_limit_P_sv_mask = (P_sv >= P_sv_max) & (dP_sv_dt > 0)
                dP_sv_dt[upper_limit_P_sv_mask] = 0
                P_sv[upper_limit_P_sv_mask] = P_sv_max
                #stop training if P_sv or V_r is out of bounds 
                if torch.any(P_sv < P_sv_min) or torch.any(P_sv > P_sv_max) or torch.any(V_r < V_r_min) or torch.any(V_r > V_r_max):
                    print(min(P_sv), max(P_sv), min(V_r), max(V_r))
                    raise ValueError("P_sv or V_r out of bounds")
                """
                return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt, dP_m_dt, dP_sv_dt]#4.116 recheck it after the meeting
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt]

        if self.model_flag == "SM6_AVR" or self.model_flag == "SM6_AVR_GOV":
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            dE_d_dash_dash_dt = (1 / self.T_q_dash_dash) * (E_d_dash - E_d_dash_dash - I_q * (self.X_q_dash - self.X_q_dash_dash))
            dE_q_dash_dash_dt = (1 / self.T_d_dash_dash) * (E_q_dash - E_q_dash_dash + I_d * (self.X_d_dash - self.X_d_dash_dash))
            dR_F_dt      = (1 / self.T_F) * (-R_F + (self.K_F / self.T_F) * E_fd)
            dV_r_dt      = (1 / self.T_A) * (-V_r + (self.K_A * R_F) - (self.K_A * self.K_F / self.T_F) * E_fd + self.K_A * (self.V_ref - V_t))
            dE_fd_dt     = (1 / self.T_E) * (-(self.K_E + 0.098 * np.e**(E_fd*0.55)) * E_fd + V_r)
            if self.model_flag=="SM6_AVR_GOV":        # Governor equations # recheck it after the meeting
                dP_m_dt  = (1 / self.T_ch) * (-P_m  + P_sv)
                dP_sv_dt = (1 / self.T_sv) * (-P_sv + self.P_c - (1/self.R_d) * omega)
                return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dE_d_dash_dash_dt, dE_q_dash_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt, dP_m_dt, dP_sv_dt]
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dE_d_dash_dash_dt, dE_q_dash_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt]
        



    def odequations_v2(self, t, x):
        """
        Calculates the derivatives of the state variables for the synchronous machine model.

        Parameters:
            t (float): The current time.
            x (list): A list of state variables, different for each model type.

        Returns:
            list: A list of derivatives, different for each model type.
        """
        if self.model_flag=="SM_IB" or self.model_flag=="SM4":
            theta, omega, E_d_dash, E_q_dash = x 
        if self.model_flag == "SM6":
            theta, omega, E_d_dash, E_q_dash, E_d_dash_dash, E_q_dash_dash = x
        if self.model_flag=="SM_AVR":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd = x
        if self.model_flag=="SM_AVR_GOV":
            theta, omega, E_d_dash, E_q_dash, R_F, V_r, E_fd, P_m, P_sv = x

        # Calculate currents from algebraic equations
        I_d, I_q = self.calculate_currents(theta, E_d_dash, E_q_dash)

        if (self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV"): # calculate V_t from algebraic equations
            V_t = self.calculate_voltages(theta, I_d, I_q)
        
        # Calculate theta derivative
        dtheta_dt = omega * self.omega_B
        
        # Calculate omega derivative
        if self.model_flag=="SM_AVR_GOV": # calculate omega derivative from algebraic equations
            domega_dt = (1/ (2 * self.H)) * (P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega * self.omega_B)
        else:
            domega_dt = (1 / (2 * self.H)) * (self.P_m - E_d_dash * I_d - E_q_dash * I_q - (self.X_q_dash - self.X_d_dash) * I_q * I_d - self.D * omega * self.omega_B)
        
        # Calculate E_dash derivatives
        if self.model_flag=="SM_IB":
            dE_q_dash_dt = 0
            dE_d_dash_dt = 0
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]
        
        if self.model_flag=="SM4":
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + self.E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt]

        if self.model_flag == "SM6":
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + self.E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            dE_q_dash_dash_dt = (1 / self.T_d_dash_dash) * (E_q_dash - E_q_dash_dash + I_d * (self.X_d_dash - self.X_d_dash_dash))
            dE_d_dash_dash_dt = (1 / self.T_q_dash_dash) * (E_d_dash - E_d_dash_dash - I_q * (self.X_q_dash - self.X_q_dash_dash))
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dE_d_dash_dash_dt, dE_q_dash_dash_dt]
        # Automatic Voltage Regulator (AVR) dynamics 
        # Exciter and AVR equations
        if (self.model_flag=="SM_AVR" or self.model_flag=="SM_AVR_GOV"):
            dE_q_dash_dt = (1 / self.T_d_dash) * (- E_q_dash - I_d * (self.X_d - self.X_d_dash) + E_fd)
            dE_d_dash_dt = (1 / self.T_q_dash) * (- E_d_dash + I_q * (self.X_q - self.X_q_dash))
            dR_F_dt      = (1 / self.T_F) * (-R_F + (self.K_F / self.T_F) * E_fd)
            dV_r_dt      = (1 / self.T_A) * (-V_r + (self.K_A * R_F) - (self.K_A * self.K_F / self.T_F) * E_fd + self.K_A * (self.V_ref - V_t))
            dE_fd_dt     = (1 / self.T_E) * (-(self.K_E + 0.098 * np.e**(E_fd*0.55)) * E_fd + V_r)
            
            if self.model_flag=="SM_AVR_GOV":        # Governor equations 
                dP_m_dt  = (1 / self.T_ch) * (-P_m  + P_sv)
                dP_sv_dt = (1 / self.T_sv) * (-P_sv + self.P_c - (1/self.R_d) * omega)
                return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt, dP_m_dt, dP_sv_dt]
            return [dtheta_dt, domega_dt, dE_d_dash_dt, dE_q_dash_dt, dR_F_dt, dV_r_dt, dE_fd_dt]
        
    

    
