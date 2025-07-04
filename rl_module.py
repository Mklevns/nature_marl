# File: marlcomm/rl_module.py
#!/usr/bin/env python3

"""
Nature-Inspired Communication RLModule for Multi-Agent Reinforcement Learning

This module implements biological communication patterns inspired by:
- Pheromone trails (chemical signaling)
- Neural plasticity (memory formation)
- Bee waggle dance (information encoding)
"""

import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from typing import Dict, Any
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.typing import TensorType


class NatureInspiredCommModule(TorchRLModule):
    """
    Custom RLModule implementing nature-inspired communication patterns.

    Features:
    - Pheromone-like communication signals (bounded chemical messaging)
    - Neural plasticity memory (GRU-based experience retention)
    - Adaptive behavior combining observations with communication history
    """

    def __init__(self, observation_space, action_space, *, model_config=None, **kwargs):
        """
        Initialize the nature-inspired communication module.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            model_config: Model configuration dictionary
            **kwargs: Additional parameters (inference_only, learner_only, etc.)
        """
        # Important: Use the new API stack constructor signature
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config or {},
            **kwargs
        )

    def setup(self):
        """Setup neural network components during initialization."""
        # Get dimensions from spaces
        obs_dim = self.observation_space.shape[0]
        if isinstance(self.action_space, Discrete):
            action_dim = self.action_space.n
        else:
            action_dim = self.action_space.shape[0]

        # Communication and memory dimensions
        self.comm_dim = 8  # Size of pheromone-like signals
        self.memory_dim = 16  # Neural memory capacity

        # Observation encoder (sensory processing)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),  # Stabilize training
        )

        # Communication signal generator (pheromone production)
        self.comm_encoder = nn.Sequential(
            nn.Linear(32, self.comm_dim),
            nn.Tanh()  # Bounded signals like chemical concentrations
        )

        # Communication signal decoder (pheromone detection)
        self.comm_decoder = nn.Sequential(
            nn.Linear(self.comm_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        # Neural plasticity memory (experience integration)
        self.memory_update = nn.GRUCell(32 + self.comm_dim, self.memory_dim)

        # Policy network (decision making)
        self.policy_net = nn.Sequential(
            nn.Linear(32 + self.memory_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),  # Prevent overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        # Value network (outcome prediction)
        self.value_net = nn.Sequential(
            nn.Linear(32 + self.memory_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Initialize persistent memory state
        self.register_buffer('hidden_state', torch.zeros(1, self.memory_dim))

    def _forward(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        """
        Core forward pass implementing nature-inspired communication.

        Args:
            batch: Input batch containing observations
            **kwargs: Additional forward arguments

        Returns:
            Dictionary with action distribution inputs and value predictions
        """
        obs = batch["obs"]
        batch_size = obs.shape[0]

        # Expand hidden state to match batch size if needed
        if self.hidden_state.shape[0] != batch_size:
            self.hidden_state = self.hidden_state.expand(batch_size, -1).contiguous()

        # Sensory processing: encode environmental observations
        obs_encoded = self.obs_encoder(obs)

        # Communication signal generation (pheromone production)
        comm_signal = self.comm_encoder(obs_encoded)

        # Communication signal processing (pheromone detection)
        # In real multi-agent setting, this would process signals from other agents
        comm_processed = self.comm_decoder(comm_signal)

        # Neural plasticity: update memory with new experience
        try:
            memory_input = torch.cat([obs_encoded, comm_signal], dim=1)
            self.hidden_state = self.memory_update(memory_input, self.hidden_state)
        except RuntimeError as e:
            # Handle tensor size mismatches gracefully
            if "size mismatch" in str(e).lower():
                # Reset memory state and try again
                self.hidden_state = torch.zeros(batch_size, self.memory_dim,
                                              device=obs.device, dtype=obs.dtype)
                self.hidden_state = self.memory_update(memory_input, self.hidden_state)
            else:
                raise e

        # Integrate processed information for decision making
        integrated_features = torch.cat([obs_encoded, self.hidden_state], dim=1)

        # Generate action distribution parameters and value estimates
        action_logits = self.policy_net(integrated_features)
        value_pred = self.value_net(integrated_features).squeeze(-1)

        return {
            "action_dist_inputs": action_logits,  # For action sampling
            "vf_preds": value_pred,  # For value function training
            "comm_signal": comm_signal,  # For potential analysis/logging
        }

    def get_initial_state(self) -> Dict[str, TensorType]:
        """Get initial state for recurrent processing."""
        return {"hidden_state": torch.zeros(1, self.memory_dim)}

    def is_stateful(self) -> bool:
        """Indicate this module maintains internal state."""
        return True


def create_nature_comm_module_spec(obs_space, act_space, model_config=None):
    """
    Factory function to create a SingleAgentRLModuleSpec for nature-inspired communication.

    Args:
        obs_space: Observation space for a single agent.
        act_space: Action space for a single agent.
        model_config: Optional model configuration.

    Returns:
        A SingleAgentRLModuleSpec configured for the NatureInspiredCommModule.
    """
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    return RLModuleSpec(
        module_class=NatureInspiredCommModule,
        observation_space=obs_space,
        action_space=act_space,
        model_config_dict=model_config or {}
    )
