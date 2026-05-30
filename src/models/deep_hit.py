"""
DeepHit: Discrete-time survival model handling competing risks.

Reference: Lee et al. 2018 — "DeepHit: A Deep Learning Approach to Survival
Analysis with Competing Risks"
https://ojs.aaai.org/index.php/AAAI/article/view/11842
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines.utils import concordance_index


class DeepHitNet(nn.Module):
    """DeepHit network: shared trunk + cause-specific output heads.

    Outputs a discrete-time probability mass function (PMF) over time bins
    for each competing risk. The PMF is obtained by applying softmax
    independently per risk over time bins.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    num_time_bins : int
        Number of discrete time intervals.
    num_risks : int
        Number of competing risks (1 for standard single-risk survival).
    hidden_dim : int
        Width of the shared hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_time_bins: int = 50,
        num_risks: int = 1,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_risks = num_risks
        self.num_time_bins = num_time_bins

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        # One output head per risk
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, num_time_bins) for _ in range(num_risks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return discrete PMF over time bins for each risk.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, input_dim)

        Returns
        -------
        torch.Tensor  shape (batch, num_risks, num_time_bins)
            Softmax-normalised probabilities per risk, per time bin.
        """
        h = self.shared(x)
        # Stack head outputs: (batch, num_risks, num_time_bins)
        logits = torch.stack([head(h) for head in self.heads], dim=1)
        return F.softmax(logits, dim=2)


class DeepHitTrainer:
    """Training wrapper for DeepHitNet.

    Parameters
    ----------
    model : DeepHitNet
    alpha : float
        Weight on log-likelihood loss vs ranking loss.  L = alpha * NLL + (1-alpha) * Rank
    sigma : float
        Bandwidth parameter for the ranking loss Gaussian kernel.
    lr : float
        Adam learning rate.
    """

    def __init__(
        self,
        model: DeepHitNet,
        alpha: float = 0.2,
        sigma: float = 0.1,
        lr: float = 1e-3,
    ) -> None:
        self.model = model
        self.alpha = alpha
        self.sigma = sigma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _digitize_times(
        self, time: np.ndarray, num_bins: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map continuous times to discrete bin indices.

        Returns
        -------
        bin_edges : np.ndarray  shape (num_bins + 1,)
        bin_idx   : np.ndarray  shape (n,)  values in [0, num_bins - 1]
        """
        t_min, t_max = float(time.min()), float(time.max())
        bin_edges = np.linspace(t_min, t_max + 1e-6, num_bins + 1)
        bin_idx = np.digitize(time, bin_edges[1:]).clip(0, num_bins - 1)
        return bin_edges, bin_idx

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def combined_loss(
        self,
        pred: torch.Tensor,
        time_bins: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined DeepHit loss.

        L = alpha * NLL_loss + (1 - alpha) * ranking_loss

        Parameters
        ----------
        pred : torch.Tensor  shape (batch, num_risks, num_time_bins) — softmax PMF
        time_bins : torch.Tensor  shape (batch,)  — discrete bin index for each patient
        event : torch.Tensor  shape (batch,)  — 0=censored, 1..num_risks=event cause

        Returns
        -------
        torch.Tensor  scalar
        """
        batch_size = pred.shape[0]
        num_risks = pred.shape[1]
        num_bins = pred.shape[2]
        device = pred.device

        # ---- 1. Log-likelihood loss ----
        # For each patient with an observed event (cause k at bin t):
        #   NLL += -log P(T=t, K=k)
        nll = torch.tensor(0.0, device=device)
        event_mask = event > 0
        if event_mask.any():
            ev_pred = pred[event_mask]          # (n_ev, num_risks, num_bins)
            ev_bins = time_bins[event_mask]     # (n_ev,)
            ev_cause = (event[event_mask] - 1).long().clamp(0, num_risks - 1)  # (n_ev,)

            # Gather probability at observed (cause, time_bin) for each event patient
            gathered = ev_pred[
                torch.arange(ev_pred.shape[0], device=device),
                ev_cause,
                ev_bins.long(),
            ]  # (n_ev,)
            nll = -torch.log(gathered.clamp(min=1e-8)).mean()

        # ---- 2. Ranking loss ----
        # For any uncensored pair (i, j) where T_i < T_j and E_i = 1:
        #   push P(T_i < T_j) to be higher (margin-based via eta)
        # We approximate with a differentiable surrogate.
        rank_loss = torch.tensor(0.0, device=device)
        if event_mask.sum() > 1:
            # CIF approximation: cumulative sum of PMF across time bins (risk 0)
            cif = pred[:, 0, :].cumsum(dim=1)  # (batch, num_bins)

            # Pairwise: eta_ij = sum_{t <= T_i} [CIF_j(t) - CIF_i(t)] for T_i < T_j
            time_bins_f = time_bins.float()
            # Vectorised: difference matrix of observed times
            di = time_bins_f.unsqueeze(1)  # (batch, 1)
            dj = time_bins_f.unsqueeze(0)  # (1, batch)
            # Pairs where patient i has event and T_i < T_j
            pair_mask = event_mask.unsqueeze(1) & (di < dj)  # (batch, batch)
            if pair_mask.any():
                # Survival difference at T_i (cumulative hazard surrogate)
                t_idx = time_bins.long().clamp(0, num_bins - 1)
                cif_at_ti = cif[
                    torch.arange(batch_size, device=device), t_idx
                ]  # (batch,)
                # eta_ij = cif_j(t_i) - cif_i(t_i)  →  want negative (j should survive longer)
                eta = (
                    cif_at_ti.unsqueeze(0) - cif_at_ti.unsqueeze(1)
                )  # (batch, batch)
                # Gaussian kernel surrogate
                rank_terms = pair_mask.float() * torch.exp(-eta / self.sigma)
                n_pairs = pair_mask.float().sum().clamp(min=1)
                rank_loss = rank_terms.sum() / n_pairs

        return self.alpha * nll + (1 - self.alpha) * rank_loss

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 64,
    ) -> list[float]:
        """Train the DeepHit model.

        Parameters
        ----------
        X : np.ndarray  shape (n, input_dim)
        time : np.ndarray  shape (n,)  — observed times
        event : np.ndarray  shape (n,)  — 0=censored, 1=event (cause 1), 2=competing (cause 2)
        n_epochs : int
        batch_size : int

        Returns
        -------
        list[float]  per-epoch training loss
        """
        self.model.train()
        device = next(self.model.parameters()).device

        num_bins = self.model.num_time_bins
        _, bin_idx = self._digitize_times(time, num_bins)
        self._bin_edges_time = (float(time.min()), float(time.max()))
        self._num_bins = num_bins

        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        T_t = torch.tensor(bin_idx, dtype=torch.long).to(device)
        E_t = torch.tensor(event, dtype=torch.float32).to(device)

        n = X_t.shape[0]
        loss_history: list[float] = []

        for _ in range(n_epochs):
            idx = torch.randperm(n)
            epoch_losses: list[float] = []
            for start in range(0, n, batch_size):
                batch_idx = idx[start : start + batch_size]
                if len(batch_idx) < 2:
                    continue
                Xb, Tb, Eb = X_t[batch_idx], T_t[batch_idx], E_t[batch_idx]

                self.optimizer.zero_grad()
                pred = self.model(Xb)
                loss = self.combined_loss(pred, Tb, Eb)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(float(loss.item()))

            loss_history.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))

        return loss_history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_survival(
        self,
        X: np.ndarray,
        time_points: list[int],
    ) -> np.ndarray:
        """Predict survival probability at given time points.

        Survival S(t) = 1 - CIF(t) = 1 - sum_{s <= t} PMF(s)  (risk 0 only).

        Parameters
        ----------
        X : np.ndarray  shape (n, input_dim)
        time_points : list[int]  — requested time points (in original time units)

        Returns
        -------
        np.ndarray  shape (n, len(time_points))  — values in [0, 1]
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            pmf = self.model(X_t)  # (n, num_risks, num_bins)

        pmf_np = pmf[:, 0, :].cpu().numpy()  # (n, num_bins)
        cif_np = np.cumsum(pmf_np, axis=1)   # (n, num_bins)

        # Map requested time_points to bin indices
        t_min, t_max = self._bin_edges_time
        num_bins = self._num_bins
        bin_edges = np.linspace(t_min, t_max + 1e-6, num_bins + 1)

        survival = np.zeros((X.shape[0], len(time_points)), dtype=np.float32)
        for j, tp in enumerate(time_points):
            bin_j = int(np.digitize(tp, bin_edges[1:])) - 1
            bin_j = int(np.clip(bin_j, 0, num_bins - 1))
            survival[:, j] = np.clip(1.0 - cif_np[:, bin_j], 0.0, 1.0)

        return survival

    def concordance_index(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
    ) -> float:
        """Compute Harrell's C-index using predicted CIF at median time.

        Parameters
        ----------
        X : np.ndarray
        time : np.ndarray
        event : np.ndarray  — 0=censored, 1=event of interest

        Returns
        -------
        float  C-index in [0, 1]
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            pmf = self.model(X_t)  # (n, num_risks, num_bins)

        pmf_np = pmf[:, 0, :].cpu().numpy()
        # Use mean CIF across all bins as risk score (higher CIF = higher risk)
        risk = pmf_np.sum(axis=1)  # scalar risk per patient

        event_binary = (event > 0).astype(int)
        return float(concordance_index(time, -risk, event_binary))
