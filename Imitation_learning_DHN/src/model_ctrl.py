#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict

# REN implementation in the acyclic version
# See paper: "Recurrent Equilibrium Networks: Flexible dynamic models with guaranteed stability and robustness"
class PsiU(nn.Module):
    def __init__(self, n, m, n_xi, l, std_ini_param=None):
        super().__init__()
        self.n = n
        self.n_xi = n_xi
        self.l = l
        self.m = m
        if std_ini_param is None:
            std_ini_param = 0.1
        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = std_ini_param
        self.X = nn.Parameter((torch.randn(2*n_xi+l, 2*n_xi+l)*std))
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi)*std))
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n)*std))
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi)*std))
        self.D21 = nn.Parameter((torch.randn(m, l)*std))
        self.D22 = nn.Parameter((torch.randn(m, n)*std))
        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n)*std))
        # bias:
        # self.bxi = nn.Parameter(torch.randn(n_xi))
        # self.bv = nn.Parameter(torch.randn(l))
        self.bu = nn.Parameter(torch.randn(m))
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi)
        self.B1 = torch.zeros(n_xi, l)
        self.E = torch.zeros(n_xi, n_xi)
        self.Lambda = torch.ones(l)
        self.C1 = torch.zeros(l, n_xi)
        self.D11 = torch.zeros(l, l)
        self.set_model_param()

    def set_model_param(self):
        n_xi = self.n_xi
        l = self.l
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2*n_xi+l)
        h1, h2, h3 = torch.split(H, [n_xi, l, n_xi], dim=0)
        H11, H12, H13 = torch.split(h1, [n_xi, l, n_xi], dim=1)
        H21, H22, _ = torch.split(h2, [n_xi, l, n_xi], dim=1)
        H31, H32, H33 = torch.split(h3, [n_xi, l, n_xi], dim=1)
        P = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + P + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, t, w, xi):
        vec = torch.zeros(self.l)
        vec[0] = 1
        epsilon = torch.zeros(self.l)
        v = F.linear(xi, self.C1[0,:]) + F.linear(w, self.D12[0,:])  # + self.bv[0]
        epsilon = epsilon + vec * torch.tanh(v/self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l)
            vec[i] = 1
            v = F.linear(xi, self.C1[i,:]) + F.linear(epsilon, self.D11[i,:]) + F.linear(w, self.D12[i,:])  # self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v/self.Lambda[i])
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon, self.B1) + F.linear(w, self.B2)  # + self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + F.linear(w, self.D22)  + self.bu
        u = 8 + (12.8 - 8) * torch.sigmoid(u)
        return u, xi_

class RenG(nn.Module):
    # ## Implementation of REN model, modified from "Recurrent Equilibrium Networks: Flexible Dynamic Models with
    # Guaranteed Stability and Robustness" by Max Revay et al.
    def __init__(self, m, p, n, l, bias=False, mode="l2stable", gamma=1, Q=None, R=None, S=None,
                 device=torch.device('cpu')):
        super().__init__()
        self.m = m  # input dimension
        self.n = n  # state dimension
        self.l = l  # dimension of v(t) and w(t)
        self.p = p  # output dimension
        self.mode = mode
        self.device = device
        self.gamma = gamma
        # # # # # # # # # IQC specification # # # # # # # # #
        self.Q = Q
        self.R = R
        self.S = S
        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = 1

        # Sparse training matrix parameters
        self.x0 = nn.Parameter((torch.randn(1, n, device=device) * std))
        self.X = nn.Parameter((torch.randn(2 * n + l, 2 * n + l, device=device) * std))
        self.Y = nn.Parameter((torch.randn(n, n, device=device) * std))
        self.Z3 = nn.Parameter(torch.randn(abs(p - m), min(p, m), device=device) * std)
        self.X3 = nn.Parameter(torch.randn(min(p, m), min(p, m), device=device) * std)
        self.Y3 = nn.Parameter(torch.randn(min(p, m), min(p, m), device=device) * std)
        self.D12 = nn.Parameter(torch.randn(l, m, device=device))
        #self.D21 = nn.Parameter((torch.randn(p, l, device=device) * std))
        self.B2 = nn.Parameter((torch.randn(n, m, device=device) * std))
        self.C2 = nn.Parameter((torch.randn(p, n, device=device) * std))

        if bias:
            self.bx = nn.Parameter(torch.randn(n, device=device) * std)
            self.bv = nn.Parameter(torch.randn(l, device=device) * std)
            self.bu = nn.Parameter(torch.randn(p, device=device) * std)
        else:
            self.bx = torch.zeros(n, device=device)
            self.bv = torch.zeros(l, device=device)
            self.bu = torch.zeros(p, device=device)
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements

        self.x = torch.zeros(1, n, device=device)
        self.epsilon = 0.001
        self.F = torch.zeros(n, n, device=device)
        self.B1 = torch.zeros(n, l, device=device)
        self.E = torch.zeros(n, n, device=device)
        self.Lambda = torch.ones(l, device=device)
        self.C1 = torch.zeros(l, n, device=device)
        self.D11 = torch.zeros(l, l, device=device)
        self.D22 = torch.zeros(p, m, device=device)
        self.P = torch.zeros(n, n, device=device)
        self.P_cal = torch.zeros(n, n, device=device)
        self.D21 = torch.zeros(p, l, device=device)
        self.set_param(gamma)

    def set_param(self, gamma=1):
        n, l, m, p = self.n, self.l, self.m, self.p
        self.Q, self.R, self.S = self._set_mode(self.mode, gamma, self.Q, self.R, self.S)
        M = F.linear(self.X3.T, self.X3.T) + self.Y3 - self.Y3.T + F.linear(self.Z3.T,
                                                                            self.Z3.T) + self.epsilon * torch.eye(
            min(m, p), device=self.device)
        if p >= m:
            N = torch.vstack((F.linear(torch.eye(m, device=self.device) - M,
                                       torch.inverse(torch.eye(m, device=self.device) + M).T),
                              -2 * F.linear(self.Z3, torch.inverse(torch.eye(m, device=self.device) + M).T)))
        else:
            N = torch.hstack((F.linear(torch.inverse(torch.eye(p, device=self.device) + M),
                                       (torch.eye(p, device=self.device) - M).T),
                              -2 * F.linear(torch.inverse(torch.eye(p, device=self.device) + M), self.Z3)))

        Lq = torch.linalg.cholesky(-self.Q).T
        Lr = torch.linalg.cholesky(self.R - torch.matmul(self.S, torch.matmul(torch.inverse(self.Q), self.S.T))).T
        self.D22 = -torch.matmul(torch.inverse(self.Q), self.S.T) + torch.matmul(torch.inverse(Lq),
                                                                                 torch.matmul(N, Lr))
        # Calculate psi_r:
        R_cal = self.R + torch.matmul(self.S, self.D22) + torch.matmul(self.S, self.D22).T + torch.matmul(self.D22.T,
                                                                                                          torch.matmul(
                                                                                                              self.Q,
                                                                                                              self.D22))
        R_cal_inv = torch.linalg.inv(R_cal)
        C2_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.C2).T
        D21_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.D21).T - self.D12
        vec_r = torch.cat((C2_cal, D21_cal, self.B2), dim=0)
        psi_r = torch.matmul(vec_r, torch.matmul(R_cal_inv, vec_r.T))
        # Calculate psi_q:
        vec_q = torch.cat((self.C2.T, self.D21.T, torch.zeros(self.n, self.p, device=self.device)), dim=0)
        psi_q = torch.matmul(vec_q, torch.matmul(self.Q, vec_q.T))
        # Create H matrix:
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n + l, device=self.device) + psi_r - psi_q
        h1, h2, h3 = torch.split(H, [n, l, n], dim=0)
        H11, H12, H13 = torch.split(h1, [n, l, n], dim=1)
        H21, H22, _ = torch.split(h2, [n, l, n], dim=1)
        H31, H32, H33 = torch.split(h3, [n, l, n], dim=1)
        self.P_cal = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + self.P_cal + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21
        # Matrix P
        self.P = torch.matmul(self.E.T, torch.matmul(torch.inverse(self.P_cal), self.E))

    def forward(self, t, u, x):
        decay_rate = 0.95
        vec = torch.zeros(self.l, device=self.device)
        epsilon = torch.zeros(self.l, device=self.device)
        if self.l > 0:
            vec[0] = 1
            v = F.linear(x, self.C1[0, :]) + F.linear(u, self.D12[0, :]) + (decay_rate ** t) * self.bv[0]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l, device=self.device)
            vec[i] = 1
            v = F.linear(x, self.C1[i, :]) + F.linear(epsilon, self.D11[i, :]) + F.linear(u, self.D12[i, :]) + (
                    decay_rate ** t) * self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[i])
        E_x_ = F.linear(x, self.F) + F.linear(epsilon, self.B1) + F.linear(u, self.B2) + (decay_rate ** t) * self.bx

        x_ = F.linear(E_x_, self.E.inverse())

        y = F.linear(x, self.C2) + F.linear(epsilon, self.D21) + F.linear(u, self.D22) + (decay_rate ** t) * self.bu
        y = 8 + (12.8 - 8) * torch.sigmoid(y)
        return y, x_

    def _set_mode(self, mode, gamma, Q, R, S, eps=1e-4):
        # We set Q to be negative definite. If Q is nsd we set: Q - \epsilon I.
        # I.e. The Q we define here is denoted as \matcal{Q} in REN paper.
        if mode == "l2stable":
            Q = -(1. / gamma) * torch.eye(self.p, device=self.device)
            R = gamma * torch.eye(self.m, device=self.device)
            S = torch.zeros(self.m, self.p, device=self.device)
        elif mode == "input_p":
            if self.p != self.m:
                raise NameError("Dimensions of u(t) and y(t) need to be the same for enforcing input passivity.")
            Q = torch.zeros(self.p, self.p, device=self.device) - eps * torch.eye(self.p, device=self.device)
            R = -2. * gamma * torch.eye(self.m, device=self.device)
            S = torch.eye(self.p, device=self.device)
        elif mode == "output_p":
            if self.p != self.m:
                raise NameError("Dimensions of u(t) and y(t) need to be the same for enforcing output passivity.")
            Q = -2. * gamma * torch.eye(self.p, device=self.device)
            R = torch.zeros(self.m, self.m, device=self.device)
            S = torch.eye(self.m, device=self.device)
        else:
            print("Using matrices R,Q,S given by user.")
            # Check dimensions:
            if not (len(R.shape) == 2 and R.shape[0] == R.shape[1] and R.shape[0] == self.m):
                raise NameError("The matrix R is not valid. It must be a square matrix of %ix%i." % (self.m, self.m))
            if not (len(Q.shape) == 2 and Q.shape[0] == Q.shape[1] and Q.shape[0] == self.p):
                raise NameError("The matrix Q is not valid. It must be a square matrix of %ix%i." % (self.p, self.p))
            if not (len(S.shape) == 2 and S.shape[0] == self.m and S.shape[1] == self.p):
                raise NameError("The matrix S is not valid. It must be a matrix of %ix%i." % (self.m, self.p))
            # Check R=R':
            if not (R == R.T).prod():
                raise NameError("The matrix R is not valid. It must be symmetric.")
            # Check Q is nsd:
            eigs, _ = torch.linalg.eig(Q)
            if not (eigs.real <= 0).prod():
                print('oh!')
                raise NameError("The matrix Q is not valid. It must be negative semidefinite.")
            if not (eigs.real < 0).prod():
                # We make Q negative definite: (\mathcal{Q} in the REN paper)
                Q = Q - eps * torch.eye(self.p, device=self.device)
        return Q, R, S


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='relu', batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state randomly
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        out = out.squeeze()
        return out

class TrajectoryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TrajectoryClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid for binary classification

    def forward(self, x):
        # Pass through LSTM layer
        lstm_out, _ = self.lstm(x)
        # Apply the linear layer and sigmoid to each time step
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out

class NoiseReconstruction(nn.Module):
    def __init__(self, f):
        super().__init__()
        n = 4
        m = 2
        self.f = f

    def forward(self, t, omega):
        x_old, u_old = omega
        x_reconstructed = self.f(t, x_old, u_old)
        return x_reconstructed


class Controller(nn.Module):
    def __init__(self, f, n, m, n_xi, l, gamma_bar, use_sp=False, t_end_sp=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.n = n
        self.m = m
        self.reconstruct_noise = NoiseReconstruction(f)
        self.ren_l2 = RenG(self.n, self.m, n_xi, l, bias=False, mode="l2stable", gamma=1)
        self.classifier = BinaryClassifier(self.n, 64)

    def forward(self, t, x_, xi, omega,gamma_bar):
        psi_x = self.reconstruct_noise(t, omega)
        w_ = x_ - psi_x
        u_t, xi_ = self.ren_l2(t, w_, xi)
        delta = self.classifier(w_)
        u_ = u_t
        omega_ = (x_, u_)
        return u_, xi_, omega_, delta



## ---------------------------------------------------------------------------------



# Robust REN implementation in the acyclic version
class REN(nn.Module):
    # ## Implementation of REN model, modified from "Recurrent Equilibrium Networks: Flexible Dynamic Models with
    # Guaranteed Stability and Robustness" by Max Revay et al.
    def __init__(self, dim_in: int, dim_out: int, dim_internal: int,
                 dim_nl: int, initialization_std: float = 0.5, internal_state_init=None, gammat=None, mode="l2stable",
                 Q=None, R=None, S=None
                 , posdef_tol: float = 0.001):
        super().__init__()

        # set dimensions
        self.dim_in = dim_in  # input dimension m
        self.dim_internal = dim_internal  # state dimension n
        self.dim_nl = dim_nl  # dimension of v(t) and w(t) l
        self.dim_out = dim_out  # output dimension p

        self.mode = mode
        self.gammat = gammat
        self.epsilon = posdef_tol
        # # # # # # # # # IQC specification # # # # # # # # #
        self.Q = Q
        self.R = R
        self.S = S
        # # # # # # # # # Training parameters # # # # # # # # #
        # Sparse training matrix parameters
        # define matrices shapes
        self.X_shape = (2 * dim_internal + dim_nl, 2 * dim_internal + dim_nl)
        self.Y_shape = (dim_internal, dim_internal)
        # nn state dynamics
        self.B2_shape = (dim_internal, dim_in)
        # nn output
        self.C2_shape = (dim_out, dim_internal)
        #self.D21_shape = (self.dim_out, self.dim_l)
        self.D22_shape = (dim_out, dim_in)
        # v signal
        self.D12_shape = (dim_nl, dim_in)
        self.Z3_shape = (abs(dim_out - dim_in), min(dim_out, dim_in))
        self.X3_shape = (min(dim_out, dim_in), min(dim_out, dim_in))
        self.Y3_shape = (min(dim_out, dim_in), min(dim_out, dim_in))
        self.gamma_shape = (1, 1)

        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'Z3', 'X3', 'Y3', 'D12']

        # Optionally define a trainable gamma
        if self.gammat is None:
            self.training_param_names.append('gamma')
        else:
            self.gamma = gammat

        # define trainable params

        self._init_trainable_params(initialization_std)

        # # # # # # # # # Non-trainable parameters and constant tensors # # # # # # # # #
        # masks
        self.register_buffer('eye_mask_min', torch.eye(min(dim_in, dim_out)))
        self.register_buffer('eye_mask_dim_in', torch.eye(dim_in))
        self.register_buffer('eye_mask_dim_out', torch.eye(dim_out))
        self.register_buffer('eye_mask_dim_state', torch.eye(dim_internal))
        self.register_buffer('eye_mask_H', torch.eye(2 * dim_internal + dim_nl))
        self.register_buffer('zeros_mask_S', torch.zeros(dim_in, dim_out))
        self.register_buffer('zeros_mask_Q', torch.zeros(dim_out, dim_out))
        self.register_buffer('zeros_mask_R', torch.zeros(dim_in, dim_in))
        self.register_buffer('zeros_mask_so', torch.zeros(dim_internal, dim_out))
        self.register_buffer('eye_mask_w', torch.eye(dim_nl))
        self.register_buffer('D21', torch.zeros(dim_out, dim_nl))

        # initialize internal state
        if internal_state_init is None:
            self.x = torch.zeros(1, 1, self.dim_internal, device="cpu")
        else:
            assert isinstance(internal_state_init, torch.Tensor)
            self.x = internal_state_init.reshape(1, 1, self.dim_internal)
        self.register_buffer('init_x', self.x.detach().clone())

        # Auxiliary elements
        self.set_param()

    def set_param(self, gamman=None):
        if gamman is not None:
            self.gamma = gamman
        gamma = torch.abs(self.gamma)
        dim_internal, dim_nl, dim_in, dim_out = self.dim_internal, self.dim_nl, self.dim_in, self.dim_out

        # Updating of Q,S,R with variable gamma if needed
        self.Q, self.R, self.S = self._set_mode(self.mode, gamma, self.Q, self.R, self.S)
        M = F.linear(self.X3.T, self.X3.T) + self.Y3 - self.Y3.T + F.linear(self.Z3.T,
                                                                            self.Z3.T) + self.epsilon * self.eye_mask_min
        if dim_out >= dim_in:
            N = torch.vstack((F.linear(self.eye_mask_dim_in - M,
                                       torch.inverse(self.eye_mask_dim_in + M).T),
                              -2 * F.linear(self.Z3, torch.inverse(self.eye_mask_dim_in + M).T)))
        else:
            N = torch.hstack((F.linear(torch.inverse(self.eye_mask_dim_out + M),
                                       (self.eye_mask_dim_out - M).T),
                              -2 * F.linear(torch.inverse(self.eye_mask_dim_out + M), self.Z3)))

        Lq = torch.linalg.cholesky(-self.Q).T
        Lr = torch.linalg.cholesky(self.R - torch.matmul(self.S, torch.matmul(torch.inverse(self.Q), self.S.T))).T
        self.D22 = -torch.matmul(torch.inverse(self.Q), self.S.T) + torch.matmul(torch.inverse(Lq),
                                                                                 torch.matmul(N, Lr))
        # Calculate psi_r:
        R_cal = self.R + torch.matmul(self.S, self.D22) + torch.matmul(self.S, self.D22).T + torch.matmul(self.D22.T,
                                                                                                          torch.matmul(
                                                                                                              self.Q,
                                                                                                              self.D22))
        R_cal_inv = torch.inverse(R_cal)
        C2_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.C2)
        D21_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.D21) - self.D12.T
        vec_r = torch.cat((C2_cal.T, D21_cal.T, self.B2), dim=0)
        psi_r = torch.matmul(vec_r, torch.matmul(R_cal_inv, vec_r.T))
        # Calculate psi_q:
        vec_q = torch.cat((self.C2.T, self.D21.T, self.zeros_mask_so), dim=0)
        psi_q = torch.matmul(vec_q, torch.matmul(self.Q, vec_q.T))
        # Create H matrix:
        H = torch.matmul(self.X.T, self.X) + self.epsilon * self.eye_mask_H + psi_r - psi_q
        h1, h2, h3 = torch.split(H, [dim_internal, dim_nl, dim_internal], dim=0)
        H11, H12, H13 = torch.split(h1, [dim_internal, dim_nl, dim_internal], dim=1)
        H21, H22, _ = torch.split(h2, [dim_internal, dim_nl, dim_internal], dim=1)
        H31, H32, H33 = torch.split(h3, [dim_internal, dim_nl, dim_internal], dim=1)
        self.P_cal = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + self.P_cal + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21
        # Matrix P
        #self.P = torch.matmul(self.E.T, torch.matmul(torch.inverse(self.P_cal), self.E))

    def forward(self, u):
        self.set_param()
        decay_rate = 0.95
        batch_size = u.shape[0]
        w = torch.zeros(batch_size, 1, self.dim_nl, device=u.device)
        # update each row of w using Eq. (8) with a lower triangular D11
        for i in range(self.dim_nl):
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u, self.D12[i, :])
            w = w + (self.eye_mask_w[i, :] * torch.tanh(v / self.Lambda[i])).reshape(batch_size, 1, self.dim_nl)

        # compute next state using Eq. 18
        self.x = F.linear(
            F.linear(self.x, self.F) + F.linear(w, self.B1) + F.linear(u, self.B2),
            self.E.inverse())

        # compute output
        y = F.linear(self.x, self.C2) + F.linear(w, self.D21) + F.linear(u, self.D22)
        y = 8 + (12.8 - 8) * torch.sigmoid(y)

        return y

    def _set_mode(self, mode, gamma, Q, R, S, eps: float = 1e-4):
        # We set Q to be negative definite. If Q is nsd we set: Q - \epsilon I.
        # I.e. The Q we define here is denoted as \matcal{Q} in REN paper.
        if mode == "l2stable":
            Q = -(1. / gamma) * self.eye_mask_dim_out
            R = gamma * self.eye_mask_dim_in
            S = self.zeros_mask_S
        elif mode == "input_p":
            if self.p != self.m:
                raise NameError("Dimensions of u(t) and y(t) need to be the same for enforcing input passivity.")
            Q = self.zeros_mask_Q - eps * self.eye_mask_dim_out
            R = -2. * gamma * self.eye_mask_dim_state
            S = self.eye_mask_dim_out
        elif mode == "output_p":
            if self.p != self.m:
                raise NameError("Dimensions of u(t) and y(t) need to be the same for enforcing output passivity.")
            Q = -2. * gamma * self.eye_mask_dim_out
            R = self.zeros_mask_R
            S = self.eye_mask_dim_state
        else:
            print("Using matrices R,Q,S given by user.")
            # Check dimensions:
            if not (len(R.shape) == 2 and R.shape[0] == R.shape[1] and R.shape[0] == self.m):
                raise NameError("The matrix R is not valid. It must be a square matrix of %ix%i." % (self.m, self.m))
            if not (len(Q.shape) == 2 and Q.shape[0] == Q.shape[1] and Q.shape[0] == self.p):
                raise NameError("The matrix Q is not valid. It must be a square matrix of %ix%i." % (self.p, self.p))
            if not (len(S.shape) == 2 and S.shape[0] == self.m and S.shape[1] == self.p):
                raise NameError("The matrix S is not valid. It must be a matrix of %ix%i." % (self.m, self.p))
            # Check R=R':
            if not (R == R.T).prod():
                raise NameError("The matrix R is not valid. It must be symmetric.")
            # Check Q is nsd:
            eigs, _ = torch.linalg.eig(Q)
            if not (eigs.real <= 0).prod():
                print('oh!')
                raise NameError("The matrix Q is not valid. It must be negative semidefinite.")
            if not (eigs.real < 0).prod():
                # We make Q negative definite: (\mathcal{Q} in the REN paper)
                Q = Q - eps * self.eye_mask_dim_out
        return Q, R, S

        # init trainable params

    def _init_trainable_params(self, initialization_std):
        for training_param_name in self.training_param_names:  # name of one of the training params, e.g., X
            # read the defined shapes of the selected training param, e.g., X_shape
            shape = getattr(self, training_param_name + '_shape')
            # define the selected param (e.g., self.X) as nn.Parameter
            if training_param_name == 'gamma':
                initialization_std = 3
            setattr(self, training_param_name, nn.Parameter((torch.randn(*shape) * initialization_std)))

        # setters and getters

    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in self.training_param_names
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in self.training_param_names
        )
        return param_dict


