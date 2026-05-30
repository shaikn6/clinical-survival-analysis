"""
DeepSurv: Neural Cox Proportional Hazard model.

Reference: Katzman et al. 2018 — "DeepSurv: personalized treatment recommender
system using a Cox proportional hazards deep neural network"
https://doi.org/10.1186/s12874-018-0482-1
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index


class DeepSurvNet(nn.Module):
    """Multi-layer perceptron outputting a log hazard ratio per patient.

    Architecture: input → [Linear → BatchNorm → ReLU → Dropout] × n_hidden → Linear(1)

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list[int]
        Width of each hidden layer.
    dropout : float
        Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [64, 32],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log hazard ratio for each patient.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, input_dim)

        Returns
        -------
        torch.Tensor  shape (batch, 1)
        """
        return self.net(x)


class DeepSurvTrainer:
    """Training wrapper for DeepSurvNet.

    Parameters
    ----------
    model : DeepSurvNet
    lr : float
        Adam learning rate.
    l2_reg : float
        L2 weight decay (AdamW weight_decay equivalent).
    """

    def __init__(
        self,
        model: DeepSurvNet,
        lr: float = 1e-3,
        l2_reg: float = 1e-4,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=l2_reg
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def cox_partial_likelihood_loss(
        self,
        log_hz: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """Breslow approximation of the Cox negative log partial likelihood.

        Parameters
        ----------
        log_hz : torch.Tensor  shape (n,)  — log hazard ratio per patient
        time : torch.Tensor    shape (n,)  — observed time
        event : torch.Tensor   shape (n,)  — 1 = event, 0 = censored

        Returns
        -------
        torch.Tensor  scalar loss
        """
        # Sort by descending time to compute risk sets efficiently
        order = torch.argsort(time, descending=True)
        log_hz = log_hz[order]
        event = event[order]

        # log-sum-exp over risk set (cumulative from top in descending order)
        log_cumsum_hz = torch.logcumsumexp(log_hz, dim=0)

        # Only sum over actual events (uncensored patients)
        event_mask = event.bool()
        neg_log_lik = -(log_hz[event_mask] - log_cumsum_hz[event_mask]).mean()
        return neg_log_lik

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 64,
    ) -> list[float]:
        """Train the model.

        Parameters
        ----------
        X_train : np.ndarray  shape (n, input_dim)
        time_train : np.ndarray  shape (n,)
        event_train : np.ndarray  shape (n,)
        n_epochs : int
        batch_size : int

        Returns
        -------
        list[float]  per-epoch training loss
        """
        self.model.train()
        device = next(self.model.parameters()).device

        X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        T_t = torch.tensor(time_train, dtype=torch.float32).to(device)
        E_t = torch.tensor(event_train, dtype=torch.float32).to(device)

        n = X_t.shape[0]
        loss_history: list[float] = []

        for _ in range(n_epochs):
            # Mini-batch sampling
            idx = torch.randperm(n)
            epoch_losses: list[float] = []
            for start in range(0, n, batch_size):
                batch_idx = idx[start : start + batch_size]
                if len(batch_idx) < 2:
                    continue
                Xb = X_t[batch_idx]
                Tb = T_t[batch_idx]
                Eb = E_t[batch_idx]
                # Skip batches with no events
                if Eb.sum() == 0:
                    continue

                self.optimizer.zero_grad()
                log_hz = self.model(Xb).squeeze(1)
                loss = self.cox_partial_likelihood_loss(log_hz, Tb, Eb)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(float(loss.item()))

            loss_history.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))

        return loss_history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (higher = higher risk).

        Parameters
        ----------
        X : np.ndarray  shape (n, input_dim)

        Returns
        -------
        np.ndarray  shape (n,)
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            scores = self.model(X_t).squeeze(1).cpu().numpy()
        return scores

    def concordance_index(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
    ) -> float:
        """Compute Harrell's concordance index.

        Parameters
        ----------
        X : np.ndarray
        time : np.ndarray
        event : np.ndarray

        Returns
        -------
        float  C-index in [0, 1]
        """
        risk = self.predict_risk(X)
        return float(concordance_index(time, -risk, event))
