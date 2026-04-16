"""
Volatility forecasting ensemble for Indian NSE options trading.

Combines GARCH(1,1), EGARCH, and TARCH models with an LSTM-based
directional momentum predictor to estimate fair-value option premiums
and detect volatility regimes.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Volatility regime labels
# ---------------------------------------------------------------------------

class VolatilityRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


# ---------------------------------------------------------------------------
# GARCH family models (pure-numpy, no arch dependency required)
# ---------------------------------------------------------------------------


class GARCHModel:
    """GARCH(1,1) model fitted via maximum-likelihood on log-returns.

    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    Parameters are estimated with a simple grid/numerical optimisation
    to avoid a hard dependency on the ``arch`` package.
    """

    def __init__(self) -> None:
        self.omega: float = 0.0
        self.alpha: float = 0.0
        self.beta: float = 0.0
        self._fitted: bool = False

    def fit(self, returns: np.ndarray) -> None:
        """Fit GARCH(1,1) to a return series.

        Uses a coarse grid search over (alpha, beta) with omega implied
        by the unconditional variance constraint.
        """
        returns = np.asarray(returns, dtype=np.float64)
        var_r = np.var(returns) + 1e-10

        best_ll = -np.inf
        best_params: Tuple[float, float, float] = (var_r * 0.01, 0.05, 0.90)

        for alpha in np.arange(0.01, 0.30, 0.02):
            for beta in np.arange(0.50, 0.99, 0.02):
                if alpha + beta >= 1.0:
                    continue
                omega = var_r * (1.0 - alpha - beta)
                if omega <= 0:
                    continue
                ll = self._log_likelihood(returns, omega, alpha, beta)
                if ll > best_ll:
                    best_ll = ll
                    best_params = (omega, alpha, beta)

        self.omega, self.alpha, self.beta = best_params
        self._fitted = True
        logger.info(
            "%s | GARCH(1,1) fitted: omega=%.6f alpha=%.4f beta=%.4f ll=%.2f",
            datetime.utcnow().isoformat(),
            self.omega,
            self.alpha,
            self.beta,
            best_ll,
        )

    def forecast(self, returns: np.ndarray, horizon: int = 1) -> float:
        """Forecast annualised volatility ``horizon`` steps ahead.

        Parameters
        ----------
        returns : np.ndarray
            Historical log-returns used to warm-start the variance filter.
        horizon : int
            Number of steps ahead to forecast.

        Returns
        -------
        float
            Annualised volatility forecast.
        """
        if not self._fitted:
            return float(np.std(returns) * np.sqrt(252))

        sigma2 = self._filter(returns)
        # Iterate the variance recursion forward
        for _ in range(horizon):
            sigma2 = self.omega + (self.alpha + self.beta) * sigma2
        return float(np.sqrt(sigma2 * 252))

    def _filter(self, returns: np.ndarray) -> float:
        """Run the GARCH filter and return the final conditional variance."""
        sigma2 = np.var(returns) + 1e-10
        for r in returns:
            sigma2 = self.omega + self.alpha * r * r + self.beta * sigma2
        return sigma2

    @staticmethod
    def _log_likelihood(
        returns: np.ndarray, omega: float, alpha: float, beta: float,
    ) -> float:
        """Gaussian log-likelihood for the GARCH(1,1) model."""
        T = len(returns)
        sigma2 = np.var(returns) + 1e-10
        ll = 0.0
        for t in range(T):
            if sigma2 < 1e-12:
                sigma2 = 1e-12
            ll += -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + returns[t] ** 2 / sigma2)
            sigma2 = omega + alpha * returns[t] ** 2 + beta * sigma2
        return ll


class EGARCHModel:
    """Exponential GARCH model for asymmetric volatility (far-future).

    log(sigma_t^2) = omega + alpha * |z_{t-1}| + gamma_1 * z_{t-1}
                     + beta * log(sigma_{t-1}^2)
    where z_t = r_t / sigma_t.
    """

    def __init__(self) -> None:
        self.omega: float = 0.0
        self.alpha: float = 0.0
        self.gamma1: float = 0.0
        self.beta: float = 0.0
        self._fitted: bool = False

    def fit(self, returns: np.ndarray) -> None:
        """Fit EGARCH parameters via grid search."""
        returns = np.asarray(returns, dtype=np.float64)
        log_var = np.log(np.var(returns) + 1e-10)

        best_ll = -np.inf
        best_params = (log_var * 0.01, 0.1, -0.05, 0.90)

        for alpha in np.arange(0.02, 0.30, 0.03):
            for beta in np.arange(0.50, 0.99, 0.03):
                for gamma1 in np.arange(-0.20, 0.05, 0.03):
                    omega = log_var * (1.0 - beta)
                    ll = self._log_likelihood(returns, omega, alpha, gamma1, beta)
                    if np.isfinite(ll) and ll > best_ll:
                        best_ll = ll
                        best_params = (omega, alpha, gamma1, beta)

        self.omega, self.alpha, self.gamma1, self.beta = best_params
        self._fitted = True
        logger.info(
            "%s | EGARCH fitted: omega=%.4f alpha=%.4f gamma=%.4f beta=%.4f",
            datetime.utcnow().isoformat(),
            self.omega,
            self.alpha,
            self.gamma1,
            self.beta,
        )

    def forecast(self, returns: np.ndarray, horizon: int = 5) -> float:
        """Forecast annualised volatility ``horizon`` steps ahead."""
        if not self._fitted:
            return float(np.std(returns) * np.sqrt(252))

        log_sigma2 = self._filter(returns)
        # Simple forward iteration assuming z=0
        for _ in range(horizon):
            log_sigma2 = self.omega + self.beta * log_sigma2
        return float(np.sqrt(np.exp(log_sigma2) * 252))

    def _filter(self, returns: np.ndarray) -> float:
        """Run the EGARCH filter, return final log(sigma^2)."""
        log_sigma2 = np.log(np.var(returns) + 1e-10)
        for r in returns:
            sigma = np.exp(log_sigma2 / 2.0)
            z = r / (sigma + 1e-12)
            log_sigma2 = (
                self.omega
                + self.alpha * (abs(z) - np.sqrt(2.0 / np.pi))
                + self.gamma1 * z
                + self.beta * log_sigma2
            )
        return log_sigma2

    @staticmethod
    def _log_likelihood(
        returns: np.ndarray,
        omega: float,
        alpha: float,
        gamma1: float,
        beta: float,
    ) -> float:
        T = len(returns)
        log_sigma2 = np.log(np.var(returns) + 1e-10)
        ll = 0.0
        for t in range(T):
            sigma2 = np.exp(log_sigma2)
            if sigma2 < 1e-12:
                sigma2 = 1e-12
            sigma = np.sqrt(sigma2)
            z = returns[t] / (sigma + 1e-12)
            ll += -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + z * z)
            log_sigma2 = (
                omega
                + alpha * (abs(z) - np.sqrt(2.0 / np.pi))
                + gamma1 * z
                + beta * log_sigma2
            )
        return ll


class TARCHModel:
    """Threshold ARCH (GJR-GARCH) for near-future volatility.

    sigma_t^2 = omega + alpha * r_{t-1}^2 + gamma_1 * r_{t-1}^2 * I(r<0)
                + beta * sigma_{t-1}^2
    """

    def __init__(self) -> None:
        self.omega: float = 0.0
        self.alpha: float = 0.0
        self.gamma1: float = 0.0
        self.beta: float = 0.0
        self._fitted: bool = False

    def fit(self, returns: np.ndarray) -> None:
        """Fit TARCH parameters via grid search."""
        returns = np.asarray(returns, dtype=np.float64)
        var_r = np.var(returns) + 1e-10

        best_ll = -np.inf
        best_params = (var_r * 0.01, 0.05, 0.05, 0.85)

        for alpha in np.arange(0.01, 0.20, 0.02):
            for gamma1 in np.arange(0.01, 0.20, 0.02):
                for beta in np.arange(0.50, 0.98, 0.03):
                    if alpha + gamma1 / 2.0 + beta >= 1.0:
                        continue
                    omega = var_r * (1.0 - alpha - gamma1 / 2.0 - beta)
                    if omega <= 0:
                        continue
                    ll = self._log_likelihood(returns, omega, alpha, gamma1, beta)
                    if ll > best_ll:
                        best_ll = ll
                        best_params = (omega, alpha, gamma1, beta)

        self.omega, self.alpha, self.gamma1, self.beta = best_params
        self._fitted = True
        logger.info(
            "%s | TARCH fitted: omega=%.6f alpha=%.4f gamma=%.4f beta=%.4f",
            datetime.utcnow().isoformat(),
            self.omega,
            self.alpha,
            self.gamma1,
            self.beta,
        )

    def forecast(self, returns: np.ndarray, horizon: int = 1) -> float:
        """Forecast annualised volatility ``horizon`` steps ahead."""
        if not self._fitted:
            return float(np.std(returns) * np.sqrt(252))

        sigma2 = self._filter(returns)
        for _ in range(horizon):
            sigma2 = self.omega + (self.alpha + self.gamma1 / 2.0 + self.beta) * sigma2
        return float(np.sqrt(sigma2 * 252))

    def _filter(self, returns: np.ndarray) -> float:
        sigma2 = np.var(returns) + 1e-10
        for r in returns:
            indicator = 1.0 if r < 0 else 0.0
            sigma2 = (
                self.omega
                + self.alpha * r * r
                + self.gamma1 * r * r * indicator
                + self.beta * sigma2
            )
        return sigma2

    @staticmethod
    def _log_likelihood(
        returns: np.ndarray,
        omega: float,
        alpha: float,
        gamma1: float,
        beta: float,
    ) -> float:
        T = len(returns)
        sigma2 = np.var(returns) + 1e-10
        ll = 0.0
        for t in range(T):
            if sigma2 < 1e-12:
                sigma2 = 1e-12
            ll += -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + returns[t] ** 2 / sigma2)
            indicator = 1.0 if returns[t] < 0 else 0.0
            sigma2 = (
                omega
                + alpha * returns[t] ** 2
                + gamma1 * returns[t] ** 2 * indicator
                + beta * sigma2
            )
        return ll


# ---------------------------------------------------------------------------
# LSTM directional momentum predictor
# ---------------------------------------------------------------------------


class VolatilityLSTM(nn.Module):
    """Simple LSTM that predicts directional volatility movement.

    Takes a sequence of historical volatility values and outputs a scalar
    prediction (+1 = vol increasing, -1 = vol decreasing).

    Parameters
    ----------
    input_dim : int
        Number of features per time-step (default 1 = just vol itself).
    hidden_dim : int
        LSTM hidden state size.
    num_layers : int
        Number of stacked LSTM layers.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` -- tanh-squashed directional prediction.
        """
        # h0, c0 default to zeros
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        return torch.tanh(self.fc(last_hidden))


# ---------------------------------------------------------------------------
# Volatility Forecaster (ensemble)
# ---------------------------------------------------------------------------


class VolatilityForecaster:
    """Ensemble volatility forecaster combining GARCH-family and LSTM models.

    Parameters
    ----------
    seq_len : int
        Look-back window for the LSTM.
    device : str, optional
        PyTorch device.
    """

    # Regime thresholds (annualised vol percentages)
    REGIME_THRESHOLDS: Dict[str, float] = {
        "low": 12.0,
        "normal": 20.0,
        "high": 30.0,
        # anything above 30 => extreme
    }

    def __init__(
        self,
        seq_len: int = 20,
        device: Optional[str] = None,
    ) -> None:
        self.seq_len = seq_len

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.garch = GARCHModel()
        self.egarch = EGARCHModel()
        self.tarch = TARCHModel()
        self.lstm = VolatilityLSTM(input_dim=1, hidden_dim=32, num_layers=1).to(self.device)
        self.lstm_optimizer = torch.optim.Adam(self.lstm.parameters(), lr=0.001)

        self._fitted: bool = False

        logger.info(
            "%s | VolatilityForecaster initialised (seq_len=%d, device=%s)",
            datetime.utcnow().isoformat(),
            seq_len,
            self.device,
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray, epochs: int = 50) -> None:
        """Fit all sub-models on historical return data.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of log-returns (daily or intraday).
        epochs : int
            Number of LSTM training epochs.
        """
        ts = datetime.utcnow().isoformat()
        returns = np.asarray(returns, dtype=np.float64)

        if len(returns) < self.seq_len + 5:
            logger.warning("%s | Not enough data to fit (%d points)", ts, len(returns))
            return

        # GARCH family
        self.garch.fit(returns)
        self.egarch.fit(returns)
        self.tarch.fit(returns)

        # LSTM: train on rolling windows of realised vol
        self._train_lstm(returns, epochs)
        self._fitted = True
        logger.info("%s | VolatilityForecaster fit complete", ts)

    def _train_lstm(self, returns: np.ndarray, epochs: int) -> None:
        """Train the LSTM on realised vol sequences."""
        # Compute rolling realised vol as the training signal
        window = min(5, len(returns) // 4)
        if window < 2:
            return

        rvol = pd.Series(returns).rolling(window).std().dropna().values
        if len(rvol) < self.seq_len + 1:
            return

        # Build sequences
        X_list: List[np.ndarray] = []
        Y_list: List[float] = []
        for i in range(self.seq_len, len(rvol)):
            seq = rvol[i - self.seq_len : i]
            target = 1.0 if rvol[i] > rvol[i - 1] else -1.0
            X_list.append(seq)
            Y_list.append(target)

        X = torch.FloatTensor(np.array(X_list)).unsqueeze(-1).to(self.device)
        Y = torch.FloatTensor(np.array(Y_list)).unsqueeze(-1).to(self.device)

        self.lstm.train()
        for epoch in range(epochs):
            pred = self.lstm(X)
            loss = nn.functional.mse_loss(pred, Y)
            self.lstm_optimizer.zero_grad()
            loss.backward()
            self.lstm_optimizer.step()

        self.lstm.eval()
        logger.debug(
            "%s | LSTM trained for %d epochs, final_loss=%.6f",
            datetime.utcnow().isoformat(),
            epochs,
            loss.item(),
        )

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast(
        self,
        returns: np.ndarray,
        near_horizon: int = 1,
        far_horizon: int = 5,
    ) -> Dict[str, float]:
        """Produce ensemble volatility forecasts.

        Parameters
        ----------
        returns : np.ndarray
            Recent log-return series.
        near_horizon : int
            Steps ahead for TARCH (near-future).
        far_horizon : int
            Steps ahead for EGARCH (far-future).

        Returns
        -------
        dict
            Keys: ``garch_vol, egarch_vol, tarch_vol, lstm_direction,
            ensemble_vol, fair_premium_pct``.
        """
        returns = np.asarray(returns, dtype=np.float64)

        garch_vol = self.garch.forecast(returns, horizon=near_horizon)
        egarch_vol = self.egarch.forecast(returns, horizon=far_horizon)
        tarch_vol = self.tarch.forecast(returns, horizon=near_horizon)
        lstm_dir = self._lstm_predict(returns)

        # Ensemble: weighted average (GARCH 40%, TARCH 35%, EGARCH 25%)
        ensemble_vol = 0.40 * garch_vol + 0.35 * tarch_vol + 0.25 * egarch_vol

        # Adjust by LSTM directional signal
        # If LSTM says vol is increasing, bias upward by up to 10%
        direction_adj = 1.0 + 0.10 * lstm_dir
        adjusted_vol = ensemble_vol * direction_adj

        # Fair premium estimate: approximate ATM straddle premium as
        # spot * vol * sqrt(T) / sqrt(252), assuming T=1 day
        fair_premium_pct = adjusted_vol * np.sqrt(1.0 / 252.0)

        result = {
            "garch_vol": garch_vol,
            "egarch_vol": egarch_vol,
            "tarch_vol": tarch_vol,
            "lstm_direction": lstm_dir,
            "ensemble_vol": adjusted_vol,
            "fair_premium_pct": fair_premium_pct,
        }

        logger.debug(
            "%s | Volatility forecast: garch=%.2f%% egarch=%.2f%% tarch=%.2f%% "
            "lstm_dir=%.2f ensemble=%.2f%% fair_prem=%.4f%%",
            datetime.utcnow().isoformat(),
            garch_vol,
            egarch_vol,
            tarch_vol,
            lstm_dir,
            adjusted_vol,
            fair_premium_pct,
        )
        return result

    def _lstm_predict(self, returns: np.ndarray) -> float:
        """Get LSTM directional prediction from recent returns.

        Returns a float in [-1, 1].
        """
        if len(returns) < self.seq_len:
            return 0.0

        # Compute rolling realised vol
        window = min(5, len(returns) // 4)
        if window < 2:
            return 0.0

        rvol = pd.Series(returns).rolling(window).std().dropna().values
        if len(rvol) < self.seq_len:
            return 0.0

        seq = rvol[-self.seq_len :]
        x = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1).to(self.device)

        self.lstm.eval()
        with torch.no_grad():
            pred = self.lstm(x)
        return float(pred.item())

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def detect_regime(self, returns: np.ndarray) -> VolatilityRegime:
        """Classify the current volatility regime.

        Parameters
        ----------
        returns : np.ndarray
            Recent log-return series.

        Returns
        -------
        VolatilityRegime
        """
        forecast = self.forecast(returns)
        vol = forecast["ensemble_vol"]

        if vol <= self.REGIME_THRESHOLDS["low"]:
            regime = VolatilityRegime.LOW
        elif vol <= self.REGIME_THRESHOLDS["normal"]:
            regime = VolatilityRegime.NORMAL
        elif vol <= self.REGIME_THRESHOLDS["high"]:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME

        logger.info(
            "%s | Volatility regime: %s (ensemble_vol=%.2f%%)",
            datetime.utcnow().isoformat(),
            regime.value,
            vol,
        )
        return regime
