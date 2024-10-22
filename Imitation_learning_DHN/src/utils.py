import torch

def set_params():
    # # # # # # # # Parameters # # # # # # # #
    mass = 2200  # Mass of the system (could be mass of a building or component, etc.)
    cop = 3.53  # Coefficient of performance (for heating or cooling systems)
    cp = 1.161 * 10 ** (-3)  # Specific heat capacity (J/kg*K)
    gamma = 0.99  # Discount factor or decay rate for the system
    ts = 0.25  # Sampling time for the system in seconds
    t_end = 40
    x0 = 30
    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-3
    epochs = 5000
    n_xi = 10  # \xi dimension -- number of states of REN
    l = 10  # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN
    n_traj = 20  # number of trajectories collected at each step of the learning
    std_ini = 0.2  # standard deviation of initial conditions
    gamma_bar = 100
    return mass, cop, cp, gamma, ts, t_end, x0, learning_rate, epochs, n_xi, l, n_traj, std_ini, gamma_bar


def set_params_online():
    params = set_params()
    mass, cop, cp, gamma, Ts, t_end, x0, learning_rate, epochs, n_xi, l, n_traj, std_ini, gamma_bar = params

    return mass, cop, cp, gamma, Ts, t_end, x0, learning_rate, epochs, n_xi, l, n_traj, std_ini, gamma_bar

