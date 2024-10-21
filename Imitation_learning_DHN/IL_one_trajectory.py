import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Import modules containing classes and functions needed for the simulation
from src.model_ctrl import Controller, RenG, BinaryClassifier, TrajectoryClassifier
from src.model_sys_DHN import DHN_sys
from src.utils import set_params

# Set the random seed for reproducibility
torch.manual_seed(1)
# Set the parameters and hyperparameters for the simulation
params = set_params()
mass, cop, cp, gamma, Ts, t_end, x0, learning_rate, epochs, n_xi, l, n_traj, std_ini, gamma_bar = params

# Define the system and controller models
sys = DHN_sys(mass, cop, cp, gamma, Ts)
ctl = RenG(sys.n, sys.m, n_xi, l, bias=True, mode="l2stable", gamma=1)
#classifier = BinaryClassifier(sys.n, 64)
classifier = TrajectoryClassifier(sys.n + sys.m, 64)

# Define the optimizer and its parameters
optimizer = torch.optim.Adam(list(ctl.parameters())+list(classifier.parameters()), lr=learning_rate)

#loss functions
MSE_loss_fn = nn.MSELoss()
class_loss_fn = nn.BCELoss()
# Initialize arrays to store the loss and its components for each epoch
MSE_loss_list = np.zeros(epochs)
class_loss_list = np.zeros(epochs)

# Define the trajectories
x = torch.tensor([30, 30.5592, 30.9346, 30.9283, 30.8404, 30.9151, 30.3146, 30.3242, 30.1839,
                  29.8278, 29.0401, 28.4426, 28.2452, 28.1426, 28.2249, 28.4195, 28.8023,
                  28.7731, 28.8672, 29.2627, 29.1933, 29.1318, 29.4561, 29.7477, 30.0748,
                  30.5884, 31.1643, 31.7491, 32.6984, 30.1530, 31.7218, 29.9548, 28.5270,
                  30.1479, 28.9972, 30.6628, 29.6387, 28.6599, 30.7320, 32.8471, 31.9077])
u = torch.tensor([8.7007, 8.9493, 9.2064, 9.5232, 9.8074, 9.9837,
                  10.291, 10.356, 10.397, 10.473, 10.722, 10.967,
                  11.034, 11.034, 10.772, 10.336, 9.6623, 9.3238,
                  8.8176, 8.2, 8.2, 8.5471, 8.2, 8.2,
                  8.5686, 8.2, 8.2, 9.0282, 1.7293e-13, 9.0459,
                  2.3319e-13, 4.8667e-14, 8.2, 1.5283e-24, 8.2, 8.8152e-22,
                  7.0386e-14, 8.2, 8.2, 5.6768e-27])
x_without_x0 = torch.tensor([30.5592, 30.9346, 30.9283, 30.8404, 30.9151, 30.3146, 30.3242, 30.1839,
                  29.8278, 29.0401, 28.4426, 28.2452, 28.1426, 28.2249, 28.4195, 28.8023,
                  28.7731, 28.8672, 29.2627, 29.1933, 29.1318, 29.4561, 29.7477, 30.0748,
                  30.5884, 31.1643, 31.7491, 32.6984, 30.1530, 31.7218, 29.9548, 28.5270,
                  30.1479, 28.9972, 30.6628, 29.6387, 28.6599, 30.7320, 32.8471, 31.9077])
# Assuming n_traj=1 for a single trajectory
x_train = x.unsqueeze(0).unsqueeze(-1)  # Shape: (1, t_end, sys.n)
u_train = u.unsqueeze(0).unsqueeze(-1)  # Shape: (1, t_end, sys.m)
x_without_x0_train = x_without_x0.unsqueeze(0).unsqueeze(-1)
x_u_train = torch.cat((x_without_x0_train, u_train), dim=-1)
labels = (u_train > 0.1).float()
x_test = x_train
u_test = u_train
x_u_test = x_u_train
# Assume x_data and u_data contain pre-collected state and input data.
# Shape: (n_traj, t_end, sys.n) for x_data, and (n_traj, t_end, sys.m) for u_data.
'''
x_data = torch.randn((n_traj, t_end, sys.n))  # Replace with your actual data
u_data = torch.randn((n_traj, t_end, sys.m))  # Replace with your actual data

# Split data into training and testing sets (e.g., 80% training, 20% testing)
n_train = int(0.8 * n_traj)
x_train = x_data[:n_train]
u_train = u_data[:n_train]
x_test = x_data[n_train:]
u_test = u_data[n_train:]
'''

# Training the controller using the NeurSLS algorithm
print("------------ Begin training ------------")
print(f"Problem: RH neurSLS -- t_end: {t_end} -- lr: {learning_rate:.2e} -- epochs: {epochs} -- std_ini: {std_ini:.2f}")
print(f"REN info -- n_xi: {n_xi} -- l: {l}")
print("--------- --------- ---------  ---------")

# Loop over the specified number of epochs
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    MSE_loss = 0  # Initialize loss

    # Loop over all training trajectories
    for kk in range(n_traj):
        x_traj = x_train[kk]  # State trajectory: (t_end, sys.n)
        u_traj = u_train[kk]  # Input trajectory: (t_end, sys.m)
        xi = torch.zeros(ctl.n)  # Reset REN internal state

        for t in range(t_end):
            x = x_traj[t]  # State at time t
            u = u_traj[t]  # Control input at time t
            x_prev = x_traj[t-1] if t > 0 else 0
            u_prev = u_traj[t-1] if t > 0 else 0
            if u < 0.1 or (u_prev < 0.1 and t > 0):
                break
            w = x - sys.noiseless_forward(t-1, x_prev, u_prev)
            u_pred, xi = ctl(t, w, xi)  # Predict control input

            # Compute the loss as the MSE between predicted and actual input
            MSE_loss += MSE_loss_fn(u_pred, u) / n_traj

    MSE_loss_list[epoch] = MSE_loss.item() / t_end  # Average loss over time steps

    #classifier training
    classifier.train()
    delta_pred = classifier(x_u_train)

    class_loss = class_loss_fn(delta_pred, labels)
    class_loss_list[epoch] = class_loss.item() / t_end  # Average loss over time steps

    print(f"Epoch: {epoch} --- Loss: {MSE_loss_list[epoch]:.4f} --- Classification loss: {class_loss_list[epoch]:.4f}")


    # Backpropagation and optimization step
    MSE_loss.backward()
    class_loss.backward()
    optimizer.step()
    ctl.set_param()

# Save the trained models to files
torch.save(ctl.state_dict(), "trained_models/OFFLINE_REN.pt")
torch.save(optimizer.state_dict(), "trained_models/OFFLINE_optimizer.pt")

# Plot the loss over epochs
plt.figure(figsize=(8, 4))
plt.plot(range(epochs), MSE_loss_list, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# # # # # # # # Test # # # # # # # #
print("------------ Begin testing ------------")

# Initialize arrays to store test loss and logs
test_MSE_loss = 0
x_test_log = torch.zeros_like(x_test)  # Log for predicted states
u_test_log = torch.zeros_like(u_test)  # Log for inputs
u_pred_test_log = torch.zeros_like(u_test) # Log for predicted inputs
w_test_log = torch.zeros_like(u_test)   # Log for disturbances

# Loop over all test trajectories
for kk in range(n_traj):
    x_traj = x_train[kk]  # State trajectory: (t_end, sys.n)
    u_traj = u_train[kk]  # Input trajectory: (t_end, sys.m)
    xi = torch.zeros(ctl.n)  # Reset REN internal state

    for t in range(t_end):
        x = x_traj[t]  # State at time t
        u = u_traj[t]  # Control input at time t
        x_prev = x_traj[t - 1] if t > 0 else 0
        u_prev = u_traj[t - 1] if t > 0 else 0
        if u < 0.1 or (u_prev < 0.1 and t > 0):
            break
        w = x - sys.noiseless_forward(t - 1, x_prev, u_prev)
        u_pred, xi = ctl(t, w, xi)  # Predict control input

        # Log the predictions and disturbances
        x_test_log[kk, t] = x
        u_test_log[kk, t] = u
        u_pred_test_log[kk, t] = u_pred
        w_test_log[kk, t] = w  # Log the disturbances

        # Compute the test loss as MSE between predicted and actual input
        test_MSE_loss += MSE_loss_fn(u_pred, u) / n_traj

print(f"Test Loss: {test_MSE_loss / t_end:.4f}")

delta_pred_test = classifier(x_u_test)

# Plot some trajectories for visual comparison between predicted and actual inputs
plt.figure(figsize=(10, 5))
for kk in range(min(3, n_traj)):  # Plot up to 3 test trajectories
    plt.plot(u_test_log[kk, :, 0].numpy(), label=f'Test u_actual_{kk}', linestyle='--')
    plt.plot(u_pred_test_log[kk, :, 0].detach().numpy(), label=f'Test u_pred_{kk}')

plt.xlabel('Time step')
plt.ylabel('Control Input')
plt.title('Actual vs. Predicted Control Inputs on Test Set')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the states and disturbances
plt.figure(figsize=(12, 6))
for kk in range(min(3, n_traj)):  # Plot up to 3 test trajectories
    plt.subplot(2, 1, 1)
    plt.plot(x_test[kk, :, 0].numpy(), label=f'Test State x_{kk}')
    plt.title('States over Time')
    plt.xlabel('Time step')
    plt.ylabel('State')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(w_test_log[kk, :, 0].detach().numpy(), label=f'Disturbance w_{kk}')
    plt.title('Disturbances over Time')
    plt.xlabel('Time step')
    plt.ylabel('Disturbance')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot the predicted labels and the actual labels
plt.figure(figsize=(10, 5))
for kk in range(min(3, n_traj)):  # Plot up to 3 test trajectories
    # Extract predicted probabilities and convert to binary labels (0 or 1)
    predicted_labels = (delta_pred_test[kk]).float().detach().numpy()
    actual_labels = labels[kk].detach().numpy()

    plt.plot(actual_labels, label=f'Test Actual Labels_{kk}', linestyle='--')
    plt.plot(predicted_labels, label=f'Test Predicted Labels_{kk}')
    plt.xlabel('Time step')
    plt.ylabel('Label')
    plt.title('Actual vs. Predicted Labels on Test Set')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()