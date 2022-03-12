from data_helpers import POPULATION_OF_LA
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class SEIRS_parameters:
    """Parameters:

    alpha: Virus induced average fatality rate
    beta: Probability of transmission per contact times the number of contacts per unit of time
    epsilon: Rate of progression from exposed to infectious (1/incubation period)
    gamma: Recovery rate of individual
    """

    population: int = POPULATION_OF_LA
    suceptible: int = POPULATION_OF_LA
    exposed: int = 0
    infected: int = 0
    recovered: int = 0
    dead: int = 0

    alpha: float = 0.01
    beta: float = 0.01
    epsilon: float = 0.01
    gamma: float = 0.01


# Sample usage:
#
# params = SEIRS_parameters(beta=0.05, gamma=0.01, lambda_param =0.001)
# print(params)
# params.beta = 0.025
# print(params)


def SEIRS(params):
    """Wrapper for the scipy solve_ivp routine and the
    SEIRS function/model

    params: a SEIRS_parameters dataclass"""

    dS_dt = -params.beta * params.infected * params.suceptible / params.population
    # dE_dt = - 

