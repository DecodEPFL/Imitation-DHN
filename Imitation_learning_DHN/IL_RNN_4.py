import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Import modules containing classes and functions needed for the simulation
from src.model_ctrl import Controller, RenG, TrajectoryClassifier, PsiU, RNNModel
from src.model_sys_DHN import DHN_sys
from src.utils import set_params

# Set the random seed for reproducibility
torch.manual_seed(1)
# Set the parameters and hyperparameters for the simulation
params = set_params()
mass, cop, cp, gamma, Ts, t_end, x0, learning_rate, epochs, n_xi, l, n_traj, std_ini, gamma_bar = params

# Define the system and controller models
sys = DHN_sys(mass, cop, cp, gamma, Ts)
# ctl = RenG(sys.n, sys.m, n_xi, l, bias=False, mode="l2stable", gamma=1)
ctl = PsiU(sys.n, sys.m, n_xi, l)
# Initialize the controller
# ctl = RNNController(input_size=sys.n + sys.m, hidden_size=64, output_size=sys.m)
# Set dimensions for RNN layers
idd = 1
hdd = 200
ldd = 2
odd = 1

# Initialize RNN model
RNN = RNNModel(idd, hdd, ldd, odd)

# classifier = BinaryClassifier(sys.n, 64)
classifier = TrajectoryClassifier(sys.n + sys.m, 64)

# Define the optimizer and its parameters
#optimizer = torch.optim.Adam(list(ctl.parameters()) + list(classifier.parameters()), lr=learning_rate)
optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
optimizer.zero_grad()

# loss functions
MSE_loss_fn = nn.MSELoss()
class_loss_fn = nn.BCELoss()
# Initialize arrays to store the loss and its components for each epoch
MSE_loss_list = np.zeros(epochs)
class_loss_list = np.zeros(epochs)
# Initialize arrays to store the loss and its components for each epoch
MSE_loss_list_val = np.zeros(epochs)
class_loss_list_val = np.zeros(epochs)

# Load the MATLAB files
# WITHOUT INITIAL CONDITIONS: OK WITH CLASSIFICATION, NOPE WITH REN
state_data = loadmat('X2.mat')['X_all']
input_data = loadmat('U2.mat')['U_all']
heat_demand = loadmat('heat_demand2.mat')['heat_demand_all']
# Transpose state_data and input_data to match the required dimensions
state_data = state_data.T  # Shape: (n_traj, t_end)
state_data = np.insert(state_data, 0, 30, axis=1)
state_data = state_data[:, :-1]
input_data = input_data.T  # Shape: (n_traj, t_end)

# Reshape and assign to x_dataset and u_dataset
x_dataset = torch.tensor(state_data).unsqueeze(-1).float()  # Shape: (n_traj, t_end, 1)
u_dataset = torch.tensor(input_data).unsqueeze(-1).float()  # Shape: (n_traj, t_end, 1)
x_u_dataset = torch.cat((x_dataset, u_dataset), dim=-1)
# plt.plot(u_dataset[5,:,0])

train_size = int(0.8 * n_traj)
x_train = x_dataset[:train_size] # Shape: (train_size, t_end, 1)
x_test = x_dataset[train_size:] # Shape: (n_traj - train_size, t_end, 1)

u_train = u_dataset[:train_size]
u_test = u_dataset[train_size:]

x_u_train = x_u_dataset[:train_size]
x_u_test = x_u_dataset[train_size:]

labels_train = (u_train > 0.1).float()
labels_test = (u_test > 0.1).float()

w_train = torch.zeros_like(u_train)  # Log for disturbances
w_test = torch.zeros_like(u_test)

#zero indices
mask_train = torch.ones((train_size, t_end), dtype=torch.bool)
mask_test = torch.ones((n_traj-train_size, t_end), dtype=torch.bool)

# Loop over all training trajectories
for kk in range(train_size):
    u_traj = u_train[kk]  # Input trajectory: (t_end, sys.m)
    x_traj = x_train[kk]

    for t in range(t_end):
        x = x_traj[t]  # State at time t
        u = u_traj[t]  # Control input at time t
        x_prev = x_traj[t - 1] if t > 0 else 0
        u_prev = u_traj[t - 1] if t > 0 else 0
        w = x - sys.noiseless_forward(t - 1, x_prev, u_prev)
        w_train[kk, t] = w  # Log the disturbances

        if (u < 0.1 or (u_prev < 0.1 and t > 0)):
            # Compute the loss as the MSE between predicted and actual input
            mask_train[kk, t] = False

# Loop over all test trajectories
for kk in range(n_traj-train_size):
    u_traj = u_test[kk]  # Input trajectory: (t_end, sys.m)
    x_traj = x_test[kk]

    for t in range(t_end):
        x = x_traj[t]  # State at time t
        u = u_traj[t]  # Control input at time t
        x_prev = x_traj[t - 1] if t > 0 else 0
        u_prev = u_traj[t - 1] if t > 0 else 0
        w = x - sys.noiseless_forward(t - 1, x_prev, u_prev)
        w_test[kk, t] = w  # Log the disturbances

        if (u < 0.1 or (u_prev < 0.1 and t > 0)):
            # Compute the loss as the MSE between predicted and actual input
            mask_test[kk, t] = False

# Initialize array to store loss values
LOSS = np.zeros(epochs)
# Train the RNN model
for epoch in range(epochs):
    # Adjust learning rate at certain epochs
    if epoch == epochs - epochs / 2:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
    if epoch == epochs - epochs / 6:
        learning_rate = 1.0e-4
        optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    # Get RNN output for training data
    yRNN = RNN(w_train)
    yRNN = torch.squeeze(yRNN)
    y = torch.squeeze(u_train)

    # Apply the mask to filter out unwanted values for y and yRNN
    y_filtered = y[mask_train].view(-1, 1)
    yRNN_filtered = yRNN[mask_train].view(-1, 1)

    # Calculate loss and backpropagate
    loss = MSE_loss_fn(y_filtered, yRNN_filtered)
    loss.backward()
    optimizer.step()

    # Print loss for current epoch
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss


# Get RNN output for validation data
yRNN_test = RNN(w_test)
yRNN_test = torch.squeeze(yRNN_test)
y_test = torch.squeeze(u_test)
y_filtered_test = y_test[mask_test].view(-1, 1)
yRNN_filtered_test = yRNN_test[mask_test].view(-1, 1)

# Calculate loss for validation data
loss_test = MSE_loss_fn(yRNN_filtered_test, y_filtered_test)

# Plot loss over epochs
plt.figure('8')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

# Plot output 1 for training data
plt.figure('9')
plt.plot(yRNN[0, :].detach().numpy(), label='REN')
plt.plot(y[0, :].detach().numpy(), label='y train')
plt.title("output 1 train single RNN")
plt.legend()
plt.show()

# Plot output 1 for validation data
plt.figure('10')
plt.plot(yRNN_test[0, :].detach().numpy(), label='REN val')
plt.plot(y_test[0, :].detach().numpy(), label='y val')
plt.title("output 1 val single RNN")
plt.legend()
plt.show()

i
