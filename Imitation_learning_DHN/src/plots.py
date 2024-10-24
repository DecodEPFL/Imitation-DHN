import torch
import matplotlib.pyplot as plt
import numpy as np

from src.loss_functions import f_loss_obst


def plot_trajectories(x, xbar, n_agents, text="", save=False, filename=None, T=100, obst=False, dots=False,
                      circles=False, axis=False, min_dist=1, f=5):
    fig = plt.figure(f)
    if obst:
        yy, xx = np.meshgrid(np.linspace(-3, 3.5, 120), np.linspace(-3, 3, 100))
        zz = xx * 0
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = f_loss_obst(torch.tensor([xx[i, j], yy[i, j], 0.0, 0.0]))
        z_min, z_max = np.abs(zz).min(), np.abs(zz).max()
        ax = fig.subplots()
        c = ax.pcolormesh(xx, yy, zz, cmap='Greens', vmin=z_min, vmax=z_max)
        # fig.colorbar(c, ax=ax)
    # plt.xlabel(r'$q_x$')
    # plt.ylabel(r'$q_y$')
    plt.title(text)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', '#90ee90', '#c20078']
    for i in range(n_agents):
        plt.plot(x[:T+1,4*i].detach(), x[:T+1,4*i+1].detach(), color=colors[i%12], linewidth=1)
        # plt.plot(x[T:,4*i].detach(), x[T:,4*i+1].detach(), color=colors[i%12], linestyle='dotted', linewidth=0.5)
        plt.plot(x[T:,4*i].detach(), x[T:,4*i+1].detach(), color='k', linewidth=0.125, linestyle='dotted')
    for i in range(n_agents):
        plt.plot(x[0,4*i].detach(), x[0,4*i+1].detach(), color=colors[i%12], marker='o', fillstyle='none')
        plt.plot(xbar[4*i].detach(), xbar[4*i+1].detach(), color=colors[i%12], marker='*')
    ax = plt.gca()
    if dots:
        for i in range(n_agents):
            for j in range(T):
                plt.plot(x[j, 4*i].detach(), x[j, 4*i+1].detach(), color=colors[i%12], marker='o')
    if circles:
        for i in range(n_agents):
            r = min_dist/2
            # if obst:
            #     circle = plt.Circle((x[T-1, 4*i].detach(), x[T-1, 4*i+1].detach()), r, color='tab:purple', fill=False)
            # else:
            circle = plt.Circle((x[T, 4*i].detach(), x[T, 4*i+1].detach()), r, color=colors[i%12], alpha=0.5,
                                zorder=10)
            ax.add_patch(circle)
    ax.axes.xaxis.set_visible(axis)
    ax.axes.yaxis.set_visible(axis)
    # TODO: add legend ( solid line: t<T/3 , dotted line> t>T/3, etc )
    if save:
        plt.savefig('figures/' + filename+'_'+text+'_trajectories.eps', format='eps')
    else:
        plt.show()
    return fig



#########
# This function plots the trajectories of agents' positions and velocities over time. It takes in the following arguments:
#t_end: the end time of the simulation
#n_agents: the number of agents in the simulation
#x: a tensor of shape (t_end, 4*n_agents) containing the positions and velocities of all agents over time
#u: an optional tensor of shape (t_end, 2*n_agents) containing the control inputs for all agents over time
#text: an optional string to add as a title to the plot
#save: a boolean indicating whether to save the plot as a file or display it
#filename: an optional string to use as the filename if save is True
#The function first creates a time tensor t and determines the number of subplots to create based on whether u is provided. It then creates a figure with the appropriate number of subplots and plots the positions and velocities of all agents over time in the first two subplots. If u is provided, it plots the control inputs for all agents over time in the third subplot. Finally, it adds a title to the entire figure and saves it as a file if save is True, or displays it if save is False.
def plot_traj_vs_time(t_end, n_agents, x, u=None, text="", save=False, filename=None):
    t = torch.linspace(0,t_end-1, t_end)
    if u is not None:
        p = 3
    else:
        p = 2
    plt.figure(figsize=(4*p, 4))
    plt.subplot(1, p, 1)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i])
        plt.plot(t, x[:,4*i+1])
    plt.xlabel(r'$t$')
    plt.title(r'$x(t)$')
    plt.subplot(1, p, 2)
    for i in range(n_agents):
        plt.plot(t, x[:,4*i+2])
        plt.plot(t, x[:,4*i+3])
    plt.xlabel(r'$t$')
    plt.title(r'$v(t)$')
    plt.suptitle(text)
    if p == 3:
        plt.subplot(1, 3, 3)
        for i in range(n_agents):
            plt.plot(t, u[:, 2*i])
            plt.plot(t, u[:, 2*i+1])
        plt.xlabel(r'$t$')
        plt.title(r'$u(t)$')
    if save:
        plt.savefig('figures/' + filename + '_' + text + '_x_u.eps', format='eps')
    else:
        plt.show()

def plot_losses(epochs, lossl, lossxl, lossul, losscal, lossobstl, text="", save=False, filename=None):
    t = torch.linspace(0, epochs - 1, epochs)
    plt.figure(figsize=(4 * 2, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, lossl[:])
    plt.xlabel(r'$epoch$')
    plt.title(r'$loss$')
    plt.subplot(1, 2, 2)
    plt.plot(t, lossxl[:])
    plt.xlabel(r'$epoch$')
    plt.title(r'$lossx$')

    plt.figure(figsize=(4 * 3, 4))
    plt.subplot(1, 3, 1)
    plt.plot(t, lossul[:])
    plt.xlabel(r'$epoch$')
    plt.title(r'$lossu$')
    plt.subplot(1, 3, 2)
    plt.plot(t, losscal[:])
    plt.xlabel(r'$epoch$')
    plt.title(r'$lossoa$')
    plt.suptitle(text)
    plt.subplot(1, 3, 3)
    plt.plot(t, lossobstl[:])
    plt.suptitle(text)
    plt.xlabel(r'$t$')
    plt.title(r'$lossobst$')

    if save:
        plt.savefig('figures/' + filename + '_' + text + '_x_u.eps', format='eps')
    else:
        plt.show()


