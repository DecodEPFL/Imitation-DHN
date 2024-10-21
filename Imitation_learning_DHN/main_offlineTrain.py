#!/usr/bin/env python
"""
Train an acyclic REN controller for the system of 2 robots in a corridor.
Author: Danilo Saccani (danilo.saccani@epfl.ch), modified from the original code by Clara Galimberti
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


# Import modules containing classes and functions needed for the simulation
from src.model_ctrl import Controller, RenG, InputOL
from src.model_sys import TwoRobots
from src.plots import plot_trajectories, plot_traj_vs_time, plot_losses
from src.loss_functions import f_loss_states, f_loss_u, f_loss_ca, f_loss_obst
from src.utils import calculate_collisions, set_params

# Set the random seed for reproducibility
torch.manual_seed(1)
# Set the parameters and hyperparameters for the simulation
params = set_params()
min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, \
alpha_u, alpha_ca, alpha_obst, n_xi, l, n_traj, std_ini, gamma_bar = params

# Set a flag to indicate whether to use an open-loop signal w = w +uOL for the controller
use_OL_signal = True

# Define the system and controller models
sys = TwoRobots(xbar,linear)
ctl = RenG(sys.n, sys.m, n_xi, l, bias=True, mode="l2stable", gamma=1)
OLs = InputOL(sys.n, t_end-10, active=use_OL_signal)

# Define the optimizer and its parameters
optimizer = torch.optim.Adam(list(ctl.parameters())+list(OLs.parameters()), lr=learning_rate)

# Generate random initial conditions for the trajectories
t_ext = t_end * 4

Mw = torch.zeros(sys.n, sys.n)
Mw[0, 0] = 1
Mw[1, 1] = 1
Mw[4, 4] = 1
Mw[5, 5] = 1
w_in = torch.randn((t_ext + 1, sys.n))
w_in = torch.matmul(w_in,Mw)
wmax = 0.05
decayw = 55

# Plot and log the open-loop trajectories before training
print("------- Print open loop trajectories --------")
x_log = torch.zeros((t_end, sys.n))
u = torch.zeros(sys.m)
x = x0
for t in range(t_end):
    x_log[t, :] = x
    w_sys = wmax*w_in[t,:]*np.exp(-t/decayw)
    x = sys(t, x, u, w_sys)
plot_trajectories(x_log, xbar, sys.n_agents, text="CL - before training", T=t_end, obst=alpha_obst)

t = torch.arange(0, t_ext+1, 1)
w = wmax*w_in[:,1]*torch.exp(-t/decayw)
plt.figure(figsize=(4 * 2, 4))
plt.plot(t, w)
plt.xlabel('time')
plt.title('w = 1.5*w*e^(-t/100)')
plt.show()

# # # # # # # # Train # # # # # # # #
# Train the controller using the NeurSLS algorithm
print("------------ Begin training ------------")
print("Problem: RH neurSLS -- t_end: %i" % t_end + " -- lr: %.2e" % learning_rate +
      " -- epochs: %i" % epochs + " -- n_traj: %i" % n_traj + " -- std_ini: %.2f" % std_ini)
print(" -- alpha_u: %.1f" % alpha_u + " -- alpha_ca: %i" % alpha_ca + " -- alpha_obst: %.1e" % alpha_obst)
print("REN info -- n_xi: %i" % n_xi + " -- l: %i" % l)
print("--------- --------- ---------  ---------")


# Initialize arrays to store the loss and its components for each epoch
loss_list = np.zeros(epochs)
loss_x_list = np.zeros(epochs)
loss_u_list = np.zeros(epochs)
loss_ca_list = np.zeros(epochs)
loss_obst_list = np.zeros(epochs)

# Loop over the specified number of epochs
for epoch in range(epochs):
    # Reset the gradients of the optimizer
    optimizer.zero_grad()
    # Initialize the loss and its components for this epoch
    loss_x, loss_u, loss_ca, loss_obst = 0, 0, 0, 0

    # Loop over the specified number of trajectories
    for kk in range(n_traj):
        x = x0
        w_in = torch.randn((t_ext + 1, sys.n))
        w_in = torch.matmul(w_in, Mw)
        xi = torch.zeros(ctl.n)
        w_REN = x0
        u, xi = ctl(0, w_REN, xi)
        usys = gamma_bar * u
        for t in range(t_end):
            x_ = x
            # Compute the next state and control input using the system and controller models
            w_sys = wmax*w_in[t,:]*np.exp(-t/decayw)
            x = sys(t, x, usys, w_sys)
            w_REN = x - sys.f(t, x_, usys) + OLs(t)
            u, xi = ctl(t, w_REN, xi)
            usys = gamma_bar * u
            # Compute the loss and its components for this time step
            loss_x = loss_x + f_loss_states(t, x, sys, Q)
            loss_u = loss_u + alpha_u * f_loss_u(t, usys)
            loss_ca = loss_ca + alpha_ca * f_loss_ca(x, sys, min_dist)
            if alpha_obst != 0:
                loss_obst = loss_obst + alpha_obst * f_loss_obst(x)

    # Compute the total loss for this epoch and log its components
    loss = loss_x + loss_u + loss_ca + loss_obst
    print("Epoch: %i --- Loss: %.4f ---||--- Loss x: %.2f --- " % (epoch, loss / t_end, loss_x) +
          "Loss u: %.2f --- Loss ca: %.2f --- Loss obst: %.2f" % (loss_u, loss_ca, loss_obst))
    loss_list[epoch] = loss.detach()
    loss_x_list[epoch] = loss_x.detach()
    loss_u_list[epoch] = loss_u.detach()
    loss_ca_list[epoch] = loss_ca.detach()
    loss_obst_list[epoch] = loss_obst.detach()

    # Backpropagate the loss through the controller model and update its parameters
    loss.backward(retain_graph=True)
    optimizer.step()
    ctl.set_param()


# Save the trained models to files
torch.save(ctl.state_dict(), "trained_models/OFFLINE_REN.pt")
torch.save(OLs.state_dict(), "trained_models/OFFLINE_sp.pt")
torch.save(optimizer.state_dict(), "trained_models/OFFLINE_optimizer.pt")

# # # # # # # # Print & plot results # # # # # # # #
# Plot the loss and its components over the epochs
plot_losses(epochs, loss_list, loss_x_list, loss_u_list, loss_ca_list, loss_obst_list)


w_in = torch.randn((t_ext + 1, sys.n))
w_in = torch.matmul(w_in, Mw)
# Compute the closed-loop trajectories using the trained controller and plot them
x_log = torch.zeros(t_ext, sys.n)
u_log = torch.zeros(t_ext, sys.m)
x = x0.detach()
xi = torch.zeros(ctl.n)
w_REN = x0.detach()
u, xi = ctl(0, w_REN, xi)
usys = gamma_bar * u
for t in range(t_ext):
    x_ = x
    x_log[t, :] = x.detach()
    u_log[t, :] = usys.detach()
    w_sys = wmax*w_in[t,:]*np.exp(-t/decayw)
    x = sys(t, x, usys, w_sys)
    w_REN = x - sys.f(t, x_, usys) + OLs(t)
    u, xi = ctl(t, w_REN, xi)
    usys = gamma_bar * u

plot_traj_vs_time(t_ext, sys.n_agents, x_log, u_log)

# Compute the number of collisions and print the result
n_coll = calculate_collisions(x_log, sys, min_dist)
print("Number of collisions after training: %d" % n_coll)
plot_trajectories(x_log, xbar, sys.n_agents, text="CL - after training - extended t", T=t_end, obst=alpha_obst)


