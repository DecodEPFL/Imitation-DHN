import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Import modules containing classes and functions needed for the simulation
from src.model_ctrl import RenG, TrajectoryClassifier, PsiU
from src.model_sys_DHN import DHN_sys
from src.utils import set_params

# Set the random seed for reproducibility
torch.manual_seed(1)
# Set the parameters and hyperparameters for the simulation
params = set_params()
mass, cop, cp, gamma, Ts, t_end, x0, learning_rate, epochs, n_xi, l, n_traj, std_ini, gamma_bar = params

# Define the system and controller models
sys = DHN_sys(mass, cop, cp, gamma, Ts)
ctl = RenG(sys.n, sys.m, n_xi, l, bias=False, mode="l2stable", gamma=1)
#ctl = PsiU(sys.n, sys.m, n_xi, l)
# Initialize the controller
#ctl = RNNController(input_size=sys.n + sys.m, hidden_size=64, output_size=sys.m)

classifier = TrajectoryClassifier(sys.n + sys.m, 64)

# Define the optimizer and its parameters
optimizer = torch.optim.Adam(list(ctl.parameters())+list(classifier.parameters()), lr=learning_rate)

#loss functions
MSE_loss_fn = nn.MSELoss()
class_loss_fn = nn.BCELoss()
# Initialize arrays to store the loss and its components for each epoch
MSE_loss_list = np.zeros(epochs)
class_loss_list = np.zeros(epochs)
# Initialize arrays to store the loss and its components for each epoch
MSE_loss_list_val = np.zeros(epochs)
class_loss_list_val = np.zeros(epochs)

# Load the MATLAB files
state_data = loadmat('X2.mat')['X_all']
input_data = loadmat('U2.mat')['U_all']
heat_demand = loadmat('heat_demand2.mat')['heat_demand_all']
# Transpose state_data and input_data to match the required dimensions
state_data = state_data.T  # Shape: (n_traj, t_end)
#All state trajectories have the same initial condition of 30
state_data = np.insert(state_data, 0, 30, axis=1)
state_data = state_data[:, :-1]
input_data = input_data.T  # Shape: (n_traj, t_end)

# Reshape and assign to x_dataset and u_dataset
x_dataset = torch.tensor(state_data).unsqueeze(-1).float()  # Shape: (n_traj, t_end, 1)
u_dataset = torch.tensor(input_data).unsqueeze(-1).float()  # Shape: (n_traj, t_end, 1)
x_u_dataset = torch.cat((x_dataset, u_dataset), dim=-1)

train_size = int(0.8 * n_traj)
x_train = x_dataset[:train_size]
x_test = x_dataset[train_size:]

u_train = u_dataset[:train_size]
u_test = u_dataset[train_size:]

x_u_train = x_u_dataset[:train_size]
x_u_test = x_u_dataset[train_size:]

labels_train = (u_train > 0.1).float()
labels_test = (u_test > 0.1).float()


# Training the controller using the NeurSLS algorithm
print("------------ Begin training ------------")
print(f"Problem: DHN -- t_end: {t_end} -- lr: {learning_rate:.2e} -- epochs: {epochs} -- std_ini: {std_ini:.2f}")
print(f"REN info -- n_xi: {n_xi} -- l: {l}")
print("--------- --------- ---------  ---------")
# Loop over the specified number of epochs
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    MSE_loss = 0  # Initialize loss
    u_pred_train_log = torch.zeros_like(u_train)  # Log for predicted inputs
    w_train_log = torch.zeros_like(u_train)  # Log for disturbances

    u_pred_val_log = torch.zeros_like(u_test)  # Log for predicted inputs
    w_val_log = torch.zeros_like(u_test)  # Log for disturbances

    # Loop over all training trajectories
    for kk in range(train_size):
        x_traj = x_train[kk]  # State trajectory: (t_end, sys.n)
        u_traj = u_train[kk]  # Input trajectory: (t_end, sys.m)
        xi = torch.zeros(ctl.n)
        #xi = torch.zeros(ctl.n_xi)  # Reset REN internal state

        number_non_zero_elements = 0
        for t in range(t_end):
            x = x_traj[t]  # State at time t
            u = u_traj[t]  # Control input at time t
            x_prev = x_traj[t-1] if t > 0 else 0
            u_prev = u_traj[t-1] if t > 0 else 0
            w = x - sys.noiseless_forward(t-1, x_prev, u_prev)
            u_pred, xi = ctl(t, w, xi)  # Predict control input

            # Log the predictions and disturbances
            u_pred_train_log[kk, t] = u_pred
            w_train_log[kk, t] = w  # Log the disturbances

            if not (u < 0.1 or (u_prev < 0.1 and t > 0)):
                # Compute the loss as the MSE between predicted and actual input
                number_non_zero_elements += 1
                MSE_loss += MSE_loss_fn(u_pred, u)
        MSE_loss = MSE_loss/number_non_zero_elements # Average loss over time steps

    MSE_loss = MSE_loss / train_size
    MSE_loss_list[epoch] = MSE_loss.item() / t_end


    #classifier training
    classifier.train()
    delta_pred = classifier(x_u_train)

    class_loss = class_loss_fn(delta_pred, labels_train)
    class_loss_list[epoch] = class_loss.item() / t_end / train_size  # Average loss over time steps

    print(f"Epoch: {epoch} --- Loss: {MSE_loss_list[epoch]:.4f} --- Classification loss: {class_loss_list[epoch]:.4f}")


    # Backpropagation and optimization step
    MSE_loss.backward()
    class_loss.backward()
    optimizer.step()
    ctl.set_param()
    #ctl.set_model_param()

    # Validation step
    with torch.no_grad():
        val_MSE_loss = 0
        for kk in range(n_traj-train_size):
            x_traj_val = x_test[kk]  # State trajectory: (t_end, sys.n)
            u_traj_val = u_test[kk]  # Input trajectory: (t_end, sys.m)
            xi_val = torch.zeros(ctl.n)  # Reset REN internal state for validation
            #xi_val = torch.zeros(ctl.n_xi)

            number_non_zero_elements = 0
            for t in range(t_end):
                x_val = x_traj_val[t]  # State at time t
                u_val = u_traj_val[t]  # Control input at time t
                x_prev_val = x_traj_val[t - 1] if t > 0 else 0
                u_prev_val = u_traj_val[t - 1] if t > 0 else 0
                w_val = x_val - sys.noiseless_forward(t - 1, x_prev_val, u_prev_val)
                u_pred_val, xi_val = ctl(t, w_val, xi_val)  # Predict control input

                u_pred_val_log[kk, t] = u_pred_val
                w_val_log[kk, t] = w_val  # Log the disturbances
                if not (u_val < 0.1 or (u_prev_val < 0.1 and t > 0)):
                    number_non_zero_elements += 1
                    # Compute the validation loss as MSE between predicted and actual input
                    val_MSE_loss += MSE_loss_fn(u_pred_val, u_val)
            val_MSE_loss = val_MSE_loss / number_non_zero_elements
        val_MSE_loss = val_MSE_loss / (n_traj - train_size)
        # Pass the state and input trajectories for classification
        delta_pred_val = classifier(x_u_test)
        val_class_loss = class_loss_fn(delta_pred_val, labels_test) / t_end / (n_traj-train_size)
        print(f"Validation MSE Loss: {val_MSE_loss.item() / t_end:.4f}")
        print(f"Validation classification Loss: {val_class_loss.item() / t_end:.4f}")
    if epoch % 100 == 0:
        # Select one trajectory (e.g., the first one) for plotting
        trajectory_idx = 5  # Change this to plot a different trajectory

        # Extract the logs for the selected trajectory
        u_pred_log = u_pred_train_log[trajectory_idx, :].detach().numpy()
        delta_pred_log = delta_pred[trajectory_idx, :].detach().numpy()
        u_real_log = u_train[trajectory_idx, :].detach().numpy()

        # Calculate the product of predicted labels and predicted inputs
        product = delta_pred_log * u_pred_log

        # Plot the product, predicted input, and predicted labels
        plt.figure(figsize=(12, 6))
        plt.plot(product, label='Predicted Label * Predicted Input', alpha=0.7, color='blue')
        plt.plot(u_real_log, label='Real Input', alpha=0.7, linestyle='--', color='green')
        plt.title(f"Epoch {epoch}: Product of Predicted Label and Predicted Input (Trajectory {trajectory_idx})")
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

    MSE_loss_list_val[epoch] = val_MSE_loss.item() / t_end  # Average loss over time steps
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

plt.figure(figsize=(10, 5))
plt.plot(range(epochs), MSE_loss_list_val, label='Validation MSE Loss', color='blue')
plt.title('Validation MSE Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation MSE Loss')
plt.grid(True)
plt.legend()
plt.show()



# # # # # # # # Test # # # # # # # #
print("------------ Begin testing ------------")

# Initialize arrays to store test loss and logs
test_MSE_loss = 0
u_pred_test_log = torch.zeros_like(u_train)  # Log for predicted inputs
w_test_log = torch.zeros_like(u_train)  # Log for disturbances


# Loop over all test trajectories
for kk in range(n_traj-train_size):
    x_traj = x_test[kk]  # State trajectory: (t_end, sys.n)
    u_traj = u_test[kk]  # Input trajectory: (t_end, sys.m)
    xi = torch.zeros(ctl.n)  # Reset REN internal state
    #xi = torch.zeros(ctl.n_xi)

    number_non_zero_elements = 0

    for t in range(t_end):
        x = x_traj[t]  # State at time t
        u = u_traj[t]  # Control input at time t
        x_prev = x_traj[t - 1] if t > 0 else 0
        u_prev = u_traj[t - 1] if t > 0 else 0
        w = x - sys.noiseless_forward(t - 1, x_prev, u_prev)
        u_pred, xi = ctl(t, w, xi)  # Predict control input

        # Log the predictions and disturbances
        u_pred_test_log[kk, t] = u_pred
        w_test_log[kk, t] = w  # Log the disturbances

        if not (u < 0.1 or (u_prev < 0.1 and t > 0)):
            # Compute the test loss as MSE between predicted and actual input
            number_non_zero_elements += 1
            test_MSE_loss += MSE_loss_fn(u_pred, u)
    test_MSE_loss = test_MSE_loss / number_non_zero_elements
test_MSE_loss = test_MSE_loss / (n_traj-train_size)
delta_pred_test = classifier(x_u_test)

test_class_loss = class_loss_fn(delta_pred_test, labels_test)/ t_end / (n_traj-train_size)

print(f"Test Loss: {test_MSE_loss:.4f}")
print(f"Classification Loss: {test_class_loss :.4f}")


# Plot the states and disturbances
plt.figure(figsize=(12, 12))
for kk in range(min(3, n_traj)):  # Plot up to 3 test trajectories
    plt.subplot(3, 1, 1)
    plt.plot(x_test[kk, :, 0].numpy(), label=f'Test State x_{kk}')
    plt.title('States over Time')
    plt.xlabel('Time step')
    plt.ylabel('State')
    plt.legend()
    plt.grid(True)
    '''
    plt.subplot(4, 1, 2)
    plt.plot(w_test_log[kk, :, 0].detach().numpy(), label=f'Disturbance w_{kk}')
    plt.title('Disturbances over Time')
    plt.xlabel('Time step')
    plt.ylabel('Disturbance')
    plt.legend()
    plt.grid(True)
    '''

    plt.subplot(3, 1, 2)
    plt.plot(u_test[kk, :, 0].detach().numpy(), label=f'Control input u_{kk}')
    plt.title('Input over Time')
    plt.xlabel('Time step')
    plt.ylabel('Input u')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(heat_demand[:, kk], label=f'Trajectory {kk + 1}')
    plt.title('Heat demand over Time')
    plt.xlabel('Time step')
    plt.ylabel('Heat demand')
    plt.legend()
    plt.grid(True)


plt.tight_layout()
plt.show()

# Plot the predicted labels and the actual labels
plt.figure(figsize=(10, 5))
for kk in range(n_traj - train_size):  # Plot up to 3 test trajectories

    # Extract predicted probabilities and convert to binary labels (0 or 1)
    predicted_labels = (delta_pred_test[kk]).float().detach().numpy()
    actual_labels = labels_test[kk].detach().numpy()

    plt.plot(actual_labels, label=f'Test Actual Labels_{kk}', linestyle='--')
    plt.plot(predicted_labels, label=f'Test Predicted Labels_{kk}')
    plt.xlabel('Time step')
    plt.ylabel('Label')
    plt.title('Actual vs. Predicted Labels on Test Set')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Extract the logs for the selected trajectory
    u_pred_log = u_pred_test_log[kk, :].detach().numpy()
    delta_pred_log = delta_pred_test[kk, :].detach().numpy()
    u_real_log = u_test[kk, :].detach().numpy()

    # Calculate the product of predicted labels and predicted inputs
    product = delta_pred_log * u_pred_log

    # Plot the product, predicted input, and predicted labels
    plt.figure(figsize=(12, 6))
    plt.plot(product, label='Predicted Label * Predicted Input', alpha=0.7, color='blue')
    plt.plot(u_real_log, label='Real Input', alpha=0.7, linestyle='--', color='green')
    plt.title(f": Product of Predicted Label and Predicted Input (Trajectory {kk})")
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

# Create a time vector (assuming 1 unit time step)
time = np.arange(t_end)

# Select the first three trajectories
for i in range(3):
    plt.plot(time, heat_demand[:, i], label=f'Trajectory {i + 1}')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Heat Demand')
plt.title('Heat Demand Trajectories')
plt.legend()
plt.grid()
plt.show()