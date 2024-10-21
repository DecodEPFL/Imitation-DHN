import torch
import torch.nn as nn
import torch.nn.functional as F

class DHN_sys(nn.Module):
    def __init__(self, mass, cop, cp, gamma, Ts, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        super().__init__()

        self.mass = mass
        self.cop = cop
        self.cp = cp
        self.gamma = gamma
        self.Ts = Ts
        self.n = 1
        self.m = 1

        # Define state-space model parameters
        A = self.gamma  # State transition matrix (constant, scalar for 1D system)
        B = self.cop * self.Ts / (self.mass * self.cp)  # Control input matrix (derived from system parameters)
        C = 1  # Output matrix (since it's a 1D system, this is just 1)
        D = 0  # Feed-through matrix (no direct input to output relationship)
        self.A, self.B, self.C, self.D = torch.tensor(A), torch.tensor(B), torch.tensor(C), torch.tensor(D)
    def noiseless_forward(self, t, x, u, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        f = self.A*x + self.B*u
        return f

    def forward(self, t, x, u, w):
        f = self.noiseless_forward(t, x, u) + w
        return f