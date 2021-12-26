import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Number contacts per day per person that will lead to an infection
# unknown but we can guess and then find out from the data
# For now using Usherwood et. al. guess of 0.35
BETA_INITIAL_GUESS = 0.35

# Gamma is 1/number of days of infectiousness
GAMMA_INITIAL_GUESS = 0.10


def SIR(t, y, beta, gamma):
    """"
    Function designed to be passed into scipy.integrate.solve_ivp
    """
    s, i, r = y[0], y[1], y[2]
    ds_dt = -beta * s * i
    di_dt = beta * s * i - gamma * i
    dr_dt = gamma * i
    return [ds_dt, di_dt, dr_dt]


def get_SIR(
        t_min, t_max,
        s0, i0, r0,
        beta=BETA_INITIAL_GUESS,
        gamma=GAMMA_INITIAL_GUESS
    ):
    """Wrapper for the scipy solve_ivp routine and the 
    SIR function/model.  Parameters:

    t_min, t_max: minimum and maximum time units to simulate.

    s0, i0, r0: initial guess for S, I, and R
    """

    sol = solve_ivp(
        SIR,
        [t_min, t_max],
        [s0, i0, r0],
        args=(beta, gamma),
        dense_output=True
    )
    return sol


def fit_beta_and_gamma(data,
                       beta=BETA_INITIAL_GUESS,
                       gamma=GAMMA_INITIAL_GUESS):
    """assume data is a n by 3 matrix where n is a series of data from timesteps
    0 to n-1, and the columns are the S, I, and R values as numbers from 0 to 1"""

    def objective_function(x, data):
        beta, gamma = x
        s0, i0, r0 = data[0]
        t_max = data.shape[0]
        solver = get_SIR(0, t_max, s0, i0, r0, beta=beta, gamma=gamma)
        t = np.linspace(1, t_max, t_max)
        z = solver.sol(t)
        diff = (z.T - data).flatten()
        rmse = np.sqrt(np.sum([d**2 for d in diff])/(len(diff)-1))
        return rmse

    res = minimize(objective_function, [beta, gamma], args=(data))
    beta_pred, gamma_pred = res.x
    return beta_pred, gamma_pred
