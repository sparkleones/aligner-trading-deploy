"""
Double DQN Agent (DDQN V3) for Indian NSE options trading.

Uses the online network to SELECT the best action and the target network to
EVALUATE that action's Q-value, eliminating the maximisation bias of vanilla
DQN.
"""

import logging
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.constants import (
    DDQN_ACTION_DIM,
    DDQN_BATCH_SIZE,
    DDQN_EPSILON_DECAY,
    DDQN_EPSILON_END,
    DDQN_EPSILON_START,
    DDQN_GAMMA,
    DDQN_LEARNING_RATE,
    DDQN_MEMORY_SIZE,
    DDQN_STATE_DIM,
    DDQN_TARGET_UPDATE_FREQ,
)
from strategy.environment import TradingEnvironment
from strategy.models import DuelingDQN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """Single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity circular replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = DDQN_MEMORY_SIZE) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Store a transition."""
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random mini-batch."""
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# DDQN Agent
# ---------------------------------------------------------------------------


class DDQNAgent:
    """Double DQN agent with dueling architecture and experience replay.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state vector.
    action_dim : int
        Number of discrete actions.
    lr : float
        Adam learning rate.
    gamma : float
        Discount factor.
    epsilon_start : float
        Initial exploration rate.
    epsilon_end : float
        Minimum exploration rate.
    epsilon_decay : float
        Multiplicative decay per episode.
    batch_size : int
        Mini-batch size for training.
    memory_size : int
        Replay buffer capacity.
    target_update_freq : int
        Soft-update the target network every N episodes.
    device : str, optional
        ``"cuda"`` or ``"cpu"``.  Auto-detected if not supplied.
    """

    def __init__(
        self,
        state_dim: int = DDQN_STATE_DIM,
        action_dim: int = DDQN_ACTION_DIM,
        lr: float = DDQN_LEARNING_RATE,
        gamma: float = DDQN_GAMMA,
        epsilon_start: float = DDQN_EPSILON_START,
        epsilon_end: float = DDQN_EPSILON_END,
        epsilon_decay: float = DDQN_EPSILON_DECAY,
        batch_size: int = DDQN_BATCH_SIZE,
        memory_size: int = DDQN_MEMORY_SIZE,
        target_update_freq: int = DDQN_TARGET_UPDATE_FREQ,
        device: Optional[str] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.online_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.update_target()  # sync weights

        # Optimiser & loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.memory = ReplayBuffer(capacity=memory_size)

        # Bookkeeping
        self.episodes_done: int = 0
        self.steps_done: int = 0
        self.training_losses: List[float] = []

        logger.info(
            "%s | DDQNAgent initialised on %s (state=%d, actions=%d, lr=%.5f, gamma=%.3f)",
            datetime.utcnow().isoformat(),
            self.device,
            state_dim,
            action_dim,
            lr,
            gamma,
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select an action using epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray
            Current state vector.
        explore : bool
            If ``False`` (evaluation mode), always pick the greedy action.

        Returns
        -------
        int
            Selected action index.
        """
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Replay buffer interaction
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """Sample a mini-batch and perform one gradient step.

        Returns
        -------
        float or None
            Training loss, or ``None`` if the buffer is too small.
        """
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([float(t.done) for t in batch]).unsqueeze(1).to(self.device)

        # --- Double DQN logic ---
        # Online network SELECTS the best next action
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # Target network EVALUATES that action
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # Current Q-values for taken actions
        current_q = self.online_net(states).gather(1, actions)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        self.steps_done += 1

        return loss_val

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def update_target(self, tau: float = 1.0) -> None:
        """Copy online network weights to the target network.

        Parameters
        ----------
        tau : float
            Interpolation factor.  ``tau=1.0`` is a hard copy;
            ``tau < 1`` performs Polyak averaging.
        """
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )
        logger.debug(
            "%s | Target network updated (tau=%.3f)",
            datetime.utcnow().isoformat(),
            tau,
        )

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def _decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        env: TradingEnvironment,
        n_episodes: int,
        verbose_every: int = 50,
    ) -> Dict[str, List[float]]:
        """Run the full training loop for ``n_episodes``.

        Parameters
        ----------
        env : TradingEnvironment
            The trading environment (will be ``reset()`` each episode).
        n_episodes : int
            Number of episodes to train.
        verbose_every : int
            Print summary every N episodes.

        Returns
        -------
        dict
            Keys ``episode_rewards``, ``episode_losses``, ``epsilons``.
        """
        ts = datetime.utcnow().isoformat()
        logger.info("%s | Starting training for %d episodes", ts, n_episodes)

        episode_rewards: List[float] = []
        episode_losses: List[float] = []
        epsilons: List[float] = []

        for ep in range(1, n_episodes + 1):
            state = env.reset()
            total_reward = 0.0
            ep_losses: List[float] = []

            while not env.done:
                action = self.select_action(state, explore=True)
                next_state, reward, done, info = env.step(action)

                self.store_transition(state, action, reward, next_state, done)
                loss = self.train_step()
                if loss is not None:
                    ep_losses.append(loss)

                state = next_state
                total_reward += reward

            # End of episode
            self.episodes_done += 1
            self._decay_epsilon()

            # Soft target update
            if self.episodes_done % self.target_update_freq == 0:
                self.update_target(tau=1.0)

            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
            episode_rewards.append(total_reward)
            episode_losses.append(avg_loss)
            epsilons.append(self.epsilon)

            if ep % verbose_every == 0:
                logger.info(
                    "%s | Episode %d/%d  reward=%.4f  avg_loss=%.6f  eps=%.4f  trades=%d",
                    datetime.utcnow().isoformat(),
                    ep,
                    n_episodes,
                    total_reward,
                    avg_loss,
                    self.epsilon,
                    env.total_trades,
                )

        logger.info(
            "%s | Training complete. Episodes=%d, final_eps=%.4f",
            datetime.utcnow().isoformat(),
            n_episodes,
            self.epsilon,
        )

        return {
            "episode_rewards": episode_rewards,
            "episode_losses": episode_losses,
            "epsilons": epsilons,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env: TradingEnvironment) -> Dict[str, float]:
        """Run one episode without exploration.

        Returns
        -------
        dict
            Keys ``total_reward, realised_pnl, total_trades, portfolio_value``.
        """
        state = env.reset()
        total_reward = 0.0

        while not env.done:
            action = self.select_action(state, explore=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward

        result = {
            "total_reward": total_reward,
            "realised_pnl": env.realised_pnl,
            "total_trades": float(env.total_trades),
            "portfolio_value": env.portfolio_value,
        }
        logger.info(
            "%s | Evaluation: reward=%.4f  PnL=%.2f  trades=%d  portfolio=%.2f",
            datetime.utcnow().isoformat(),
            result["total_reward"],
            result["realised_pnl"],
            int(result["total_trades"]),
            result["portfolio_value"],
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save agent state to disk.

        Parameters
        ----------
        filepath : str
            Path to the ``.pt`` checkpoint file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episodes_done": self.episodes_done,
            "steps_done": self.steps_done,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        torch.save(checkpoint, str(path))
        logger.info(
            "%s | Agent saved to %s (eps=%.4f, episodes=%d)",
            datetime.utcnow().isoformat(),
            filepath,
            self.epsilon,
            self.episodes_done,
        )

    def load(self, filepath: str) -> None:
        """Load agent state from disk.

        Parameters
        ----------
        filepath : str
            Path to the ``.pt`` checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.episodes_done = checkpoint.get("episodes_done", 0)
        self.steps_done = checkpoint.get("steps_done", 0)

        logger.info(
            "%s | Agent loaded from %s (eps=%.4f, episodes=%d)",
            datetime.utcnow().isoformat(),
            filepath,
            self.epsilon,
            self.episodes_done,
        )
