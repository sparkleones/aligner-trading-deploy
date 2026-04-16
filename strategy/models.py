"""
Neural network architectures for the DDQN trading agent.

Implements a Dueling DQN network that decomposes the Q-value into a
state-value stream V(s) and an advantage stream A(s, a).
"""

import logging
from datetime import datetime

import torch
import torch.nn as nn

from config.constants import DDQN_ACTION_DIM, DDQN_STATE_DIM

logger = logging.getLogger(__name__)


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network.

    Architecture
    ------------
    Shared backbone:
        Linear(state_dim, 256) -> ReLU -> Linear(256, 128) -> ReLU

    Value stream:
        Linear(128, 64) -> ReLU -> Linear(64, 1)

    Advantage stream:
        Linear(128, 64) -> ReLU -> Linear(64, action_dim)

    Output:
        Q(s, a) = V(s) + A(s, a) - mean_a[A(s, a)]

    Parameters
    ----------
    state_dim : int
        Dimensionality of the input state vector.
    action_dim : int
        Number of discrete actions.
    """

    def __init__(
        self,
        state_dim: int = DDQN_STATE_DIM,
        action_dim: int = DDQN_ACTION_DIM,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Value stream  V(s) -> scalar
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage stream  A(s, a) -> |A| values
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self._init_weights()
        logger.info(
            "%s | DuelingDQN created: state_dim=%d, action_dim=%d, params=%d",
            datetime.utcnow().isoformat(),
            state_dim,
            action_dim,
            sum(p.numel() for p in self.parameters()),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions given a batch of states.

        Parameters
        ----------
        state : torch.Tensor
            Shape ``(batch, state_dim)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, action_dim)`` -- Q-values.
        """
        features = self.shared(state)
        value: torch.Tensor = self.value_stream(features)          # (batch, 1)
        advantage: torch.Tensor = self.advantage_stream(features)  # (batch, action_dim)

        # Dueling aggregation: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
