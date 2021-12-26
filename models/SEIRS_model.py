import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Number contacts per day per person that will lead to an infection
# unknown but we can guess and then find out from the data
# For now using Usherwood et. al. guess of 0.35
BETA_INITIAL_GUESS = 0.35

# Gamma is 1/number of days of infectiousness
GAMMA_INITIAL_GUESS = 0.10
