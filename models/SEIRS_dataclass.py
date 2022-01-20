from dataclasses import dataclass


@dataclass
class SEIRS_parameters:
    suceptible: float = 1.0
    exposed: float = 0.0
    infected: float = 0.0
    recovered: float = 0.0
    dead: float = 0.0
    beta: float = 0.01
    gamma: float = 0.01
    lambda_param: float = 0.01


# Sample usage:
#
# params = SEIRS_parameters(beta=0.05, gamma=0.01, lambda_param =0.001)
# print(params)
# params.beta = 0.025
# print(params)
