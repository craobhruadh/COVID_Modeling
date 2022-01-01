import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Number contacts per day per person that will lead to an infection
# unknown but we can guess and then find out from the data
# For now using Usherwood et. al. guess of 0.35
BETA_INITIAL_GUESS = 0.35

# Gamma is 1/number of days of infectiousness
ALPHA_INITIAL_GUESS=0
GAMMA_INITIAL_GUESS=0.10
NU_INITIAL_GUESS=0
DELTA_INITIAL_GUESS=0

def SEIRS(t, y, beta=None, alpha=None, nu=None, gamma=None, delta=None):
    """"
    Function designed to be passed into scipy.integrate.solve_ivp

    y: list/iterable of S, E, I, and R.  Assumes it'll be four elements long
    beta: transmission rate
    alpha: 1/exposure time 
    nu: 1/time infections
    gamma: 1/time spent (temporary) immune
    delta: demographic birth and death rate

    Resources: 
    https://www.nature.com/articles/s41592-020-0856-2
    https://link.springer.com/article/10.1007/s00285-019-01374-z
    https://www.sciencedirect.com/science/article/abs/pii/S0025556402001098
    https://www.thelancet.com/action/showPdf?pii=S1473-3099%2820%2930769-6
    """
    s, e, i, r = y[0], y[1], y[2], y[3]
    ds_dt = delta-beta*s*i+gamma*r-delta*s
    de_dt = beta*s*i - (alpha+delta)*e
    di_dt = alpha*e - (nu+delta)*i
    dr_dt = nu*i - (gamma+delta) * r
    return [ds_dt, de_dt, di_dt, dr_dt]


def get_SEIRS_solver(
        t_min, t_max,
        s0, e0, i0, r0,
        beta=BETA_INITIAL_GUESS,
        alpha=ALPHA_INITIAL_GUESS,
        nu=NU_INITIAL_GUESS,
        gamma=GAMMA_INITIAL_GUESS,
        delta=DELTA_INITIAL_GUESS
    ):
    """Wrapper for the scipy solve_ivp routine and the 
    SIR function/model.  Parameters:

    t_min, t_max: minimum and maximum time units to simulate.

    s0, i0, r0: initial guess for S, I, and R
    """

    sol = solve_ivp(
        SEIRS,
        [t_min, t_max],
        [s0, e0, i0, r0],
        args=(beta, gamma),
        dense_output=True
    )
    return sol

def fit_parameters(
    data,
    beta=BETA_INITIAL_GUESS,
    alpha=ALPHA_INITIAL_GUESS,
    nu=NU_INITIAL_GUESS,
    gamma=GAMMA_INITIAL_GUESS,
    delta=DELTA_INITIAL_GUESS
):
    """assume data is a n by 3 matrix where n is a series of data from timesteps
    0 to n-1, and the columns are the S, I, and R values as numbers from 0 to 1
    
    To review:
    https://link.springer.com/article/10.1007/s00285-019-01374-z
    https://www.sciencedirect.com/science/article/abs/pii/S0025556402001098
    """

    def objective_function(x, data):
        beta, gamma = x
        s0, e0, i0, r0 = data[0]
        t_max = data.shape[0]
        solver = get_SEIRS_solver(0, t_max, s0, e0, i0, r0, beta=beta, alpha=alpha, nu=nu, gamma=gamma,delta=delta)        
        t = np.linspace(1, t_max, t_max)
        z = solver.sol(t)
        diff = (z.T - data).flatten()
        rmse = np.sqrt(np.sum([d**2 for d in diff])/(len(diff)-1))
        return rmse

    res = minimize(objective_function, [beta, alpha, nu, gamma, delta], args=(data))
    beta_pred, alpha_pred, nu_pred, gamma_pred, delta_pred = res.x

    return beta_pred, alpha_pred, nu_pred, gamma_pred, delta_pred
