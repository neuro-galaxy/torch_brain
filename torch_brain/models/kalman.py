import torch
import torch.nn as nn

class KalmanFilter(nn.Module):
    def __init__(self, state_dim, obs_dim, device='cpu'):
        """
        Kalman Filter for neural decoding (Wu et al., 2003 style).
        Args:
            state_dim (int): Dimension of the kinematic state vector.
            obs_dim (int): Dimension of the observation (neural signals).
            device (str): 'cpu' or 'cuda'
        """
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # State transition matrix (A)
        self.A = nn.Parameter(torch.eye(state_dim))
        # Observation matrix (C)
        self.C = nn.Parameter(torch.randn(obs_dim, state_dim) * 0.01)

        # Process noise covariance (Q)
        self.Q = nn.Parameter(torch.eye(state_dim) * 1e-3)
        # Observation noise covariance (R)
        self.R = nn.Parameter(torch.eye(obs_dim) * 1e-2)

    def predict(self, x_prev, P_prev):
        """Time update (prediction step)."""
        # Predicted state estimate
        x_pred = self.A @ x_prev
        # Predicted covariance estimate
        P_pred = self.A @ P_prev @ self.A.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, y_obs):
        """Measurement update (correction step)."""
        # Kalman gain
        S = self.C @ P_pred @ self.C.T + self.R  # innovation covariance
        K = P_pred @ self.C.T @ torch.linalg.inv(S)

        # Updated state estimate
        y_pred = self.C @ x_pred
        x_new = x_pred + K @ (y_obs - y_pred)

        # Updated covariance estimate
        I = torch.eye(self.state_dim, device=self.device)
        P_new = (I - K @ self.C) @ P_pred

        return x_new, P_new, K

    def forward(self, y_seq, x_init=None, P_init=None):
        """
        Run Kalman filter on a sequence of neural observations.
        Args:
            y_seq (Tensor): [T, obs_dim] sequence of observations
            x_init (Tensor): [state_dim,] initial state
            P_init (Tensor): [state_dim, state_dim] initial covariance
        Returns:
            x_filtered: [T, state_dim] filtered state estimates
        """
        T = y_seq.shape[0]
        if x_init is None:
            x_init = torch.zeros(self.state_dim, device=self.device)
        if P_init is None:
            P_init = torch.eye(self.state_dim, device=self.device)

        x_filtered = []
        x_prev, P_prev = x_init, P_init

        for t in range(T):
            # Predict
            x_pred, P_pred = self.predict(x_prev, P_prev)
            # Update with current observation
            x_new, P_new, _ = self.update(x_pred, P_pred, y_seq[t])
            # Store and iterate
            x_filtered.append(x_new)
            x_prev, P_prev = x_new, P_new

        return torch.stack(x_filtered, dim=0)

