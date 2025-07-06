# File: nature_marl/rl_module.py
#!/usr/bin/env python3

"""
Enhanced Nature-Inspired Communication RLModule for Multi-Agent Reinforcement Learning

This module implements advanced biological communication patterns inspired by:
- Pheromone trails (chemical signaling with attention mechanisms)
- Neural plasticity (adaptive memory formation)
- Bee waggle dance (multi-round information encoding)
- Swarm intelligence (collective decision making)

Key Features:
- Multi-head attention for targeted communication
- Multi-round communication protocols
- Entropy and sparsity regularization
- Centralized critic for cooperative learning
- Adaptive memory with GRU-based plasticity
"""

import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from typing import Dict, Any, Optional, Tuple
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
import numpy as np


class MultiRoundAttentionComm(nn.Module):
    """
    Multi-round attention-based communication module inspired by neural networks
    in social insects and collective decision-making processes.
    """

    def __init__(self, embed_dim: int, num_heads: int, num_rounds: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_rounds = num_rounds

        # Multi-round attention layers (like iterative bee waggle dance)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=0.1
            ) for _ in range(num_rounds)
        ])

        # Layer normalization for stable training
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_rounds)
        ])

        # Residual projections for each round
        self.residual_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_rounds)
        ])

    def forward(self, embeddings: torch.Tensor, num_agents: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform multi-round attention-based communication.

        Args:
            embeddings: Shape (batch_size * num_agents, embed_dim)
            num_agents: Number of agents in the environment

        Returns:
            Tuple of (final_embeddings, aggregated_messages)
        """
        batch_size = embeddings.shape[0] // num_agents

        # Reshape for multi-agent attention: (batch_size, num_agents, embed_dim)
        h = embeddings.view(batch_size, num_agents, -1)

        attention_weights_history = []

        # Multi-round communication (like iterative pheromone reinforcement)
        for round_idx in range(self.num_rounds):
            # Self-attention among agents
            h_attended, attention_weights = self.attention_layers[round_idx](h, h, h)
            attention_weights_history.append(attention_weights)

            # Residual connection with projection
            h_residual = self.residual_projections[round_idx](h)
            h = self.layer_norms[round_idx](h_attended + h_residual)

        # Aggregate final messages for each agent
        final_embeddings = h.view(batch_size * num_agents, -1)

        # Compute aggregated attention weights across all rounds
        avg_attention = torch.stack(attention_weights_history, dim=0).mean(dim=0)

        return final_embeddings, avg_attention


class PheromoneEncoder(nn.Module):
    """
    Pheromone-like signal encoder that creates bounded chemical-like messages.
    """

    def __init__(self, input_dim: int, pheromone_dim: int):
        super().__init__()
        self.pheromone_dim = pheromone_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, pheromone_dim),
            nn.Tanh()  # Bounded like chemical concentrations
        )

        # Learnable concentration scaling
        self.concentration_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate pheromone-like signals."""
        pheromones = self.encoder(x)
        return pheromones * self.concentration_scale


class NeuralPlasticityMemory(nn.Module):
    """
    Adaptive memory system inspired by synaptic plasticity in biological neural networks.
    """

    def __init__(self, input_dim: int, memory_dim: int):
        super().__init__()
        self.memory_dim = memory_dim

        # Multi-layer GRU for complex memory dynamics
        self.memory_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=memory_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Memory consolidation network
        self.consolidation_net = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
            nn.Sigmoid()  # Gating mechanism for memory retention
        )

    def forward(self, experience: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Update memory with new experience using neural plasticity principles.

        Args:
            experience: New sensory and communication experience
            hidden_state: Current memory state

        Returns:
            Updated memory state
        """
        # Add sequence dimension for GRU
        experience_seq = experience.unsqueeze(1)

        # Update memory through GRU dynamics
        memory_output, new_hidden = self.memory_gru(experience_seq, hidden_state)
        memory_output = memory_output.squeeze(1)  # Remove sequence dimension

        # Apply consolidation gating
        consolidation_gate = self.consolidation_net(memory_output)
        consolidated_memory = memory_output * consolidation_gate

        return consolidated_memory, new_hidden


class NatureInspiredCommModule(TorchRLModule):
    """
    Enhanced nature-inspired communication module with advanced features:
    - Multi-head attention for targeted communication
    - Multi-round communication protocols
    - Pheromone-like bounded messaging
    - Neural plasticity memory
    - Custom loss regularization
    """

    def __init__(self, observation_space, action_space, *, model_config=None, **kwargs):
        """
        Initialize the enhanced nature-inspired communication module.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            model_config: Model configuration dictionary
            **kwargs: Additional parameters (inference_only, learner_only, etc.)
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config or {},
            **kwargs
        )

    def setup(self):
        """Setup enhanced neural network components with communication capabilities."""
        # Get configuration parameters
        single_agent_obs_dim = self.observation_space.shape[0]
        self.num_agents = self.model_config.get("num_agents", 1)
        true_obs_dim = single_agent_obs_dim * self.num_agents

        if isinstance(self.action_space, Discrete):
            action_dim = self.action_space.n
        else:
            action_dim = self.action_space.shape[0]

        # Enhanced communication dimensions
        self.embed_dim = self.model_config.get("embed_dim", 256)
        self.pheromone_dim = self.model_config.get("pheromone_dim", 16)
        self.memory_dim = self.model_config.get("memory_dim", 64)
        self.num_comm_rounds = self.model_config.get("num_comm_rounds", 3)
        self.num_attention_heads = self.model_config.get("num_attention_heads", 8)

        # Network configuration
        fcnet_hiddens = self.model_config.get("fcnet_hiddens", [512, 256])
        fcnet_activation = self.model_config.get("fcnet_activation", "relu")

        # Observation encoder (enhanced sensory processing)
        self.obs_encoder = nn.Sequential(
            nn.Linear(true_obs_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        # Pheromone communication system
        self.pheromone_encoder = PheromoneEncoder(self.embed_dim, self.pheromone_dim)

        # Multi-round attention communication
        self.attention_comm = MultiRoundAttentionComm(
            embed_dim=self.embed_dim,
            num_heads=self.num_attention_heads,
            num_rounds=self.num_comm_rounds
        )

        # Neural plasticity memory system
        self.memory_system = NeuralPlasticityMemory(
            input_dim=self.embed_dim + self.pheromone_dim,
            memory_dim=self.memory_dim
        )

        # Policy network (decentralized actor)
        self.policy_net = self._create_network(
            input_dim=self.embed_dim + self.memory_dim,
            output_dim=action_dim,
            hiddens=fcnet_hiddens,
            activation=fcnet_activation
        )

        # Centralized value network (has access to all agent information)
        centralized_value_input = self.embed_dim * self.num_agents + self.memory_dim
        self.value_net = self._create_network(
            input_dim=centralized_value_input,
            output_dim=1,
            hiddens=fcnet_hiddens,
            activation=fcnet_activation
        )

        # Initialize memory states
        self.register_buffer('memory_state', torch.zeros(1, 2, self.memory_dim))  # 2 layers

        # Communication statistics for custom loss
        self.register_buffer("comm_entropy", torch.tensor(0.0))
        self.register_buffer("comm_usage", torch.tensor(0.0))
        self.register_buffer("attention_diversity", torch.tensor(0.0))

    def _create_network(self, input_dim: int, output_dim: int, hiddens: list, activation: str) -> nn.Module:
        """Create a neural network with specified architecture."""
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
        }
        activation_layer = activation_map.get(activation.lower(), nn.ReLU)

        layers = []
        current_dim = input_dim

        for h_dim in hiddens:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                activation_layer(),
                nn.Dropout(0.1)
            ])
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def _forward(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        """
        Enhanced forward pass with multi-round attention communication.

        Args:
            batch: Input batch containing observations
            **kwargs: Additional forward arguments

        Returns:
            Dictionary with action distribution inputs, value predictions, and communication stats
        """
        obs = batch["obs"]
        batch_size = obs.shape[0]

        # Expand memory state to match batch size
        if self.memory_state.shape[0] != batch_size:
            self.memory_state = self.memory_state.expand(batch_size, -1, -1).contiguous()

        # Encode observations (sensory processing)
        obs_encoded = self.obs_encoder(obs)

        # Generate pheromone signals
        pheromone_signals = self.pheromone_encoder(obs_encoded)

        # Multi-round attention communication
        if self.num_agents > 1:
            comm_enhanced_features, attention_weights = self.attention_comm(
                obs_encoded, self.num_agents
            )

            # Calculate attention diversity for regularization
            self.attention_diversity = self._calculate_attention_diversity(attention_weights)
        else:
            comm_enhanced_features = obs_encoded
            self.attention_diversity = torch.tensor(0.0, device=obs.device)

        # Neural plasticity memory update
        memory_input = torch.cat([comm_enhanced_features, pheromone_signals], dim=1)

        try:
            memory_output, new_memory_state = self.memory_system(memory_input, self.memory_state)
            self.memory_state = new_memory_state
        except RuntimeError as e:
            if "size mismatch" in str(e).lower():
                # Reset memory state and try again
                self.memory_state = torch.zeros(batch_size, 2, self.memory_dim,
                                               device=obs.device, dtype=obs.dtype)
                memory_output, self.memory_state = self.memory_system(memory_input, self.memory_state)
            else:
                raise e

        # Decentralized policy features (individual agent decision making)
        policy_features = torch.cat([comm_enhanced_features, memory_output], dim=1)
        action_logits = self.policy_net(policy_features)

        # Centralized value function features (access to all agent info)
        if self.num_agents > 1:
            # Reshape to get all agent features
            agent_features = comm_enhanced_features.view(batch_size // self.num_agents, -1)
            centralized_features = torch.cat([agent_features, memory_output[:batch_size // self.num_agents]], dim=1)
            value_pred = self.value_net(centralized_features)
            # Expand value predictions for all agents
            value_pred = value_pred.repeat(self.num_agents, 1).squeeze(-1)
        else:
            centralized_features = policy_features
            value_pred = self.value_net(centralized_features).squeeze(-1)

        # Calculate communication statistics for custom loss
        self._update_communication_stats(pheromone_signals, attention_weights if self.num_agents > 1 else None)

        return {
            "action_dist_inputs": action_logits,
            "vf_preds": value_pred,
            "pheromone_signals": pheromone_signals,
            "communication_stats": {
                "entropy": self.comm_entropy,
                "usage": self.comm_usage,
                "attention_diversity": self.attention_diversity
            }
        }

    def _calculate_attention_diversity(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Calculate diversity of attention patterns to encourage varied communication."""
        if attention_weights is None:
            return torch.tensor(0.0)

        # Calculate entropy of attention distribution
        attention_probs = torch.softmax(attention_weights, dim=-1)
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1)
        return entropy.mean()

    def _update_communication_stats(self, pheromones: torch.Tensor, attention_weights: Optional[torch.Tensor]):
        """Update communication statistics for regularization."""
        # Pheromone signal entropy (diversity)
        pheromone_probs = torch.softmax(pheromones, dim=-1)
        self.comm_entropy = -torch.sum(
            pheromone_probs * torch.log(pheromone_probs + 1e-8),
            dim=-1
        ).mean()

        # Pheromone signal usage (sparsity)
        self.comm_usage = torch.abs(pheromones).mean()

    def get_communication_loss(self, entropy_coeff: float = 0.01, sparsity_coeff: float = 0.001,
                              diversity_coeff: float = 0.01) -> torch.Tensor:
        """
        Calculate custom communication loss for regularization.

        Args:
            entropy_coeff: Coefficient for entropy bonus (encourages diversity)
            sparsity_coeff: Coefficient for sparsity penalty (encourages efficiency)
            diversity_coeff: Coefficient for attention diversity bonus

        Returns:
            Communication regularization loss
        """
        # Entropy bonus (encourage diverse communication)
        entropy_loss = -entropy_coeff * self.comm_entropy

        # Sparsity penalty (encourage efficient communication)
        sparsity_loss = sparsity_coeff * self.comm_usage

        # Attention diversity bonus (encourage varied attention patterns)
        diversity_loss = -diversity_coeff * self.attention_diversity

        total_comm_loss = entropy_loss + sparsity_loss + diversity_loss
        return total_comm_loss

    def get_initial_state(self) -> Dict[str, TensorType]:
        """Get initial state for recurrent processing."""
        return {
            "memory_state": torch.zeros(1, 2, self.memory_dim),
            "comm_entropy": torch.tensor(0.0),
            "comm_usage": torch.tensor(0.0),
            "attention_diversity": torch.tensor(0.0)
        }

    def is_stateful(self) -> bool:
        """Indicate this module maintains internal state."""
        return True


def create_enhanced_nature_comm_module_spec(obs_space, act_space, model_config=None):
    """
    Factory function to create a SingleAgentRLModuleSpec for enhanced nature-inspired communication.

    Args:
        obs_space: Observation space for a single agent.
        act_space: Action space for a single agent.
        model_config: Optional model configuration.

    Returns:
        A SingleAgentRLModuleSpec configured for the enhanced NatureInspiredCommModule.
    """
    from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec

    return SingleAgentRLModuleSpec(
        module_class=NatureInspiredCommModule,
        observation_space=obs_space,
        action_space=act_space,
        model_config_dict=model_config or {}
    )


# Configuration helpers for common setups
def get_nature_comm_config(num_agents: int = 8, embed_dim: int = 256,
                          comm_rounds: int = 3, attention_heads: int = 8) -> Dict[str, Any]:
    """
    Get a standard configuration for nature-inspired communication.

    Args:
        num_agents: Number of agents in the environment
        embed_dim: Embedding dimension for features
        comm_rounds: Number of communication rounds
        attention_heads: Number of attention heads

    Returns:
        Configuration dictionary
    """
    return {
        "num_agents": num_agents,
        "embed_dim": embed_dim,
        "pheromone_dim": embed_dim // 16,  # Typically much smaller than embeddings
        "memory_dim": embed_dim // 4,     # Memory dimension
        "num_comm_rounds": comm_rounds,
        "num_attention_heads": attention_heads,
        "fcnet_hiddens": [embed_dim * 2, embed_dim],
        "fcnet_activation": "relu"
    }
