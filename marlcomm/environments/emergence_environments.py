# File: marlcomm/emergence_environments.py
"""
Emergence Environments for Communication Research

This module implements diverse multi-agent environments designed to create
natural pressures for communication emergence. Each environment is inspired
by biological scenarios where communication evolved as a survival advantage.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv


class CommunicationChannel:
    """Base class for different communication modalities."""

    def __init__(self, channel_dim: int, max_range: float = np.inf):
        self.channel_dim = channel_dim
        self.max_range = max_range
        self.messages = {}  # agent_id -> message

    @abstractmethod
    def transmit(self, sender_id: str, message: np.ndarray,
                 sender_pos: np.ndarray) -> None:
        """Transmit a message from sender."""
        pass

    @abstractmethod
    def receive(self, receiver_id: str, receiver_pos: np.ndarray) -> np.ndarray:
        """Receive messages at receiver's position."""
        pass

    def reset(self):
        """Clear all messages."""
        self.messages.clear()


class PheromoneField(CommunicationChannel):
    """
    Spatial pheromone field that mimics ant chemical communication.
    Pheromones decay over time and diffuse through space.
    """

    def __init__(self, grid_size: Tuple[int, int], n_chemicals: int = 2,
                 decay_rate: float = 0.95, diffusion_rate: float = 0.1):
        super().__init__(n_chemicals, max_range=1.0)  # Local sensing only
        self.grid_size = grid_size
        self.field = np.zeros((*grid_size, n_chemicals))
        self.decay_rate = decay_rate
        self.diffusion_rate = diffusion_rate

    def transmit(self, sender_id: str, message: np.ndarray,
                 sender_pos: np.ndarray) -> None:
        """Deposit pheromone at sender's location."""
        x, y = int(sender_pos[0]), int(sender_pos[1])
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            self.field[x, y] += message

    def receive(self, receiver_id: str, receiver_pos: np.ndarray) -> np.ndarray:
        """Sense pheromone concentration at receiver's location."""
        x, y = int(receiver_pos[0]), int(receiver_pos[1])
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            # Sense local neighborhood
            x_min, x_max = max(0, x-1), min(self.grid_size[0], x+2)
            y_min, y_max = max(0, y-1), min(self.grid_size[1], y+2)
            local_field = self.field[x_min:x_max, y_min:y_max]
            return local_field.mean(axis=(0, 1))
        return np.zeros(self.channel_dim)

    def update(self):
        """Apply decay and diffusion to pheromone field."""
        # Decay
        self.field *= self.decay_rate

        # Diffusion (simple averaging with neighbors)
        new_field = self.field.copy()
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                neighbors = []
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                        neighbors.append(self.field[ni, nj])
                if neighbors:
                    diffused = np.mean(neighbors, axis=0) * self.diffusion_rate
                    new_field[i, j] = self.field[i, j] * (1 - self.diffusion_rate) + diffused
        self.field = new_field

    def reset(self):
        """Clear pheromone field."""
        self.field.fill(0)


class SymbolicChannel(CommunicationChannel):
    """Discrete symbolic communication like bee dance patterns."""

    def __init__(self, vocab_size: int = 16, max_message_length: int = 4,
                 broadcast_range: float = 5.0):
        super().__init__(max_message_length, broadcast_range)
        self.vocab_size = vocab_size
        self.messages = {}  # agent_id -> (message, position)

    def transmit(self, sender_id: str, message: np.ndarray,
                 sender_pos: np.ndarray) -> None:
        """Broadcast symbolic message."""
        # Convert continuous values from [-1, 1] to [0, 1] for symbol conversion
        normalized_message = (message + 1.0) / 2.0
        # Then convert to discrete symbols
        symbols = np.clip(normalized_message * self.vocab_size, 0, self.vocab_size - 1).astype(int)
        self.messages[sender_id] = (symbols, sender_pos.copy())

    def receive(self, receiver_id: str, receiver_pos: np.ndarray) -> np.ndarray:
        """Receive symbolic messages within range."""
        received = np.zeros(self.channel_dim)
        message_count = 0

        for sender_id, (message, sender_pos) in self.messages.items():
            if sender_id != receiver_id:
                distance = np.linalg.norm(receiver_pos - sender_pos)
                if distance <= self.max_range:
                    # One-hot decode and average multiple messages
                    for i, symbol in enumerate(message[:self.channel_dim]):
                        if symbol < self.vocab_size:
                            received[i] += symbol / self.vocab_size
                    message_count += 1

        if message_count > 0:
            received /= message_count

        return received


@dataclass
class Resource:
    """Resource in the environment."""
    position: np.ndarray
    value: float
    quantity: float
    resource_type: int = 0
    regeneration_rate: float = 0.0

    def harvest(self, amount: float) -> float:
        """Harvest resource and return actual amount obtained."""
        harvested = min(amount, self.quantity)
        self.quantity -= harvested
        return harvested

    def regenerate(self):
        """Regenerate resource over time."""
        self.quantity = min(self.quantity + self.regeneration_rate, self.value)


class EmergenceEnvironment(ParallelEnv, ABC):
    """Base class for environments that promote communication emergence."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "emergence_env"}

    def __init__(self, n_agents: int = 5, grid_size: Tuple[int, int] = (20, 20), **kwargs):
        super().__init__()
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.episode_length = kwargs.pop("episode_length", 200)
        self.render_mode = kwargs.pop("render_mode", None)

        # Base observation dimension for all environments
        self.base_obs_dim = 10

        # Agent properties
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_positions = {}
        self.agent_orientations = {}  # For directional movement
        self.agent_energy = {}

        # Communication
        self.communication_channel = None
        self.communication_dim = kwargs.pop("communication_dim", 0)
        self.setup_communication_channel()

        # Metrics tracking
        self.episode_metrics = {
            "communication_events": 0,
            "successful_coordination": 0,
            "resource_efficiency": 0.0,
            "information_shared": 0.0
        }

        self.current_step = 0

    @abstractmethod
    def setup_communication_channel(self):
        """Initialize the communication channel for this environment."""
        pass

    @abstractmethod
    def _get_obs(self, agent: str) -> np.ndarray:
        """Get observation for a single agent."""
        pass

    @abstractmethod
    def _get_reward(self, agent: str) -> float:
        """Calculate reward for a single agent."""
        pass

    @abstractmethod
    def _get_info(self, agent: str) -> Dict[str, Any]:
        """Get additional info for a single agent."""
        pass

    def observation_space(self, agent: str) -> spaces.Space:
        """Observation space for each agent."""
        # Base observation: position, orientation, energy, local environment
        comm_obs_dim = self.communication_channel.channel_dim if self.communication_channel else 0
        return spaces.Box(low=-1, high=1, shape=(self.base_obs_dim + comm_obs_dim,), dtype=np.float32)

    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """Dictionary of observation spaces for PettingZoo compatibility."""
        return {agent: self.observation_space(agent) for agent in self.possible_agents}

    def action_space(self, agent: str) -> spaces.Space:
        """Action space for each agent."""
        # Movement (4 directions) + communication actions
        n_movement_actions = 5  # up, down, left, right, stay
        n_comm_actions = self.communication_channel.channel_dim if self.communication_channel else 0
        return spaces.Box(low=-1, high=1, shape=(n_movement_actions + n_comm_actions,), dtype=np.float32)

    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """Dictionary of action spaces for PettingZoo compatibility."""
        return {agent: self.action_space(agent) for agent in self.possible_agents}


class ForagingEnvironment(EmergenceEnvironment):
    """
    Ant-inspired foraging environment where agents must find and collect resources.
    Communication emerges to share resource locations efficiently.
    """

    def __init__(self, n_agents: int = 5, grid_size: Tuple[int, int] = (30, 30),
                 n_food_clusters: int = 3, cluster_size: int = 5, **kwargs):
        self.n_food_clusters = n_food_clusters
        self.cluster_size = cluster_size
        self.resources: List[Resource] = []
        self.home_position = np.array(grid_size) // 2
        self.collected_resources = 0

        super().__init__(n_agents, grid_size=grid_size, **kwargs)

    def setup_communication_channel(self):
        """Use pheromone field for ant-like communication."""
        self.communication_channel = PheromoneField(
            self.grid_size,
            n_chemicals=self.communication_dim if self.communication_dim > 0 else 2,
            decay_rate=0.98,
            diffusion_rate=0.05
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.collected_resources = 0

        # Place agents near home
        for i, agent in enumerate(self.agents):
            angle = 2 * np.pi * i / len(self.agents)
            offset = np.array([np.cos(angle), np.sin(angle)]) * 3
            self.agent_positions[agent] = self.home_position + offset
            self.agent_orientations[agent] = np.random.rand() * 2 * np.pi
            self.agent_energy[agent] = 100.0

        # Create food clusters
        self.resources.clear()
        for _ in range(self.n_food_clusters):
            cluster_center = np.random.rand(2) * np.array(self.grid_size)
            for _ in range(self.cluster_size):
                offset = np.random.randn(2) * 2
                pos = np.clip(cluster_center + offset, 0, np.array(self.grid_size) - 1)
                self.resources.append(Resource(
                    position=pos,
                    value=10.0,
                    quantity=10.0,
                    regeneration_rate=0.1
                ))

        # Reset communication
        self.communication_channel.reset()

        # Reset metrics
        self.episode_metrics = {
            "communication_events": 0,
            "successful_coordination": 0,
            "resource_efficiency": 0.0,
            "information_shared": 0.0
        }

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        """Execute one environment step."""
        self.current_step += 1

        # Process agent actions
        rewards = {}
        for agent, action in actions.items():
            if agent in self.agents:
                try:
                    # Movement
                    self._process_movement(agent, action[:5])

                    # Communication (pheromone deposition)
                    if len(action) > 5:
                        comm_signal = action[5:7].copy()  # Two pheromone types
                        # Stronger pheromone when carrying food
                        if self._is_carrying_food(agent):
                            comm_signal[0] *= 2.0  # Food trail pheromone
                        self.communication_channel.transmit(
                            agent, comm_signal, self.agent_positions[agent]
                        )
                        self.episode_metrics["communication_events"] += np.sum(np.abs(comm_signal)) > 0.1

                    # Resource collection
                    self._collect_resources(agent)

                    # Energy consumption
                    self.agent_energy[agent] -= 0.5
                    if self.agent_energy[agent] <= 0:
                        self.agents.remove(agent)
                except Exception as e:
                    print(f"Error processing actions for agent {agent}: {e}")
                    # Optionally remove agent or set reward to a penalty
                    if agent in self.agents:
                        self.agents.remove(agent)

        # Update pheromone field
        if isinstance(self.communication_channel, PheromoneField):
            self.communication_channel.update()

        # Resource regeneration
        for resource in self.resources:
            resource.regenerate()

        # Calculate rewards
        rewards = {agent: self._get_reward(agent) for agent in self.agents}

        # Check termination
        terminations = {agent: agent not in self.agents for agent in self.possible_agents}
        truncations = {agent: self.current_step >= self.episode_length for agent in self.possible_agents}

        # Update metrics
        self._update_metrics()

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def _process_movement(self, agent: str, movement_action: np.ndarray):
        """Process agent movement based on action."""
        # Softmax over movement directions
        probs = np.exp(movement_action) / np.sum(np.exp(movement_action))
        direction = np.random.choice(5, p=probs)

        # Update position
        pos = self.agent_positions[agent].copy()
        if direction == 0:  # Up
            pos[1] = max(0, pos[1] - 1)
        elif direction == 1:  # Down
            pos[1] = min(self.grid_size[1] - 1, pos[1] + 1)
        elif direction == 2:  # Left
            pos[0] = max(0, pos[0] - 1)
        elif direction == 3:  # Right
            pos[0] = min(self.grid_size[0] - 1, pos[0] + 1)
        # direction == 4 is stay

        self.agent_positions[agent] = pos

    def _collect_resources(self, agent: str):
        """Check if agent can collect resources at current position."""
        pos = self.agent_positions[agent]
        for resource in self.resources:
            if np.linalg.norm(pos - resource.position) < 1.0:
                harvested = resource.harvest(1.0)
                if harvested > 0:
                    self.agent_energy[agent] += harvested * 2
                    # Check if agent is at home to deposit
                    if np.linalg.norm(pos - self.home_position) < 2.0:
                        self.collected_resources += harvested

    def _is_carrying_food(self, agent: str) -> bool:
        """Check if agent has recently collected food (simplified)."""
        return self.agent_energy[agent] > 105.0  # Has extra energy from food

    def _get_obs(self, agent: str) -> np.ndarray:
        if agent not in self.agents:
            # Use the observation_space method to get the correct shape
            return np.zeros(self.observation_space(agent).shape)

        pos = self.agent_positions[agent]

        # This observation vector MUST match the dimension defined in observation_space()
        # The base_obs_dim is 10.
        basic_obs = np.zeros(10, dtype=np.float32)
        basic_obs[0] = pos[0] / self.grid_size[0]
        basic_obs[1] = pos[1] / self.grid_size[1]
        basic_obs[2] = np.cos(self.agent_orientations[agent])
        basic_obs[3] = np.sin(self.agent_orientations[agent])
        basic_obs[4] = np.clip(self.agent_energy[agent] / 100.0, -1.0, 1.0)
        basic_obs[5] = (self.home_position[0] - pos[0]) / self.grid_size[0]
        basic_obs[6] = (self.home_position[1] - pos[1]) / self.grid_size[1]
        basic_obs[7] = 1.0 if self._detect_nearby_food(agent) else 0.0
        basic_obs[8] = len(self.agents) / self.n_agents  # Team size ratio
        basic_obs[9] = self.current_step / self.episode_length  # Time progress

        # Pheromone observation
        pheromone_obs = self.communication_channel.receive(agent, pos)

        return np.clip(np.concatenate([basic_obs, pheromone_obs]), -1.0, 1.0).astype(np.float32)

    def _detect_nearby_food(self, agent: str) -> bool:
        """Check if food is nearby (within sensing range)."""
        pos = self.agent_positions[agent]
        for resource in self.resources:
            if resource.quantity > 0 and np.linalg.norm(pos - resource.position) < 3.0:
                return True
        return False

    def _get_reward(self, agent: str) -> float:
        """Calculate reward focusing on collective foraging efficiency."""
        if agent not in self.agents:
            return 0.0

        reward = 0.0

        # Small penalty for energy consumption (encourages efficiency)
        reward -= 0.01

        # Reward for being near food
        if self._detect_nearby_food(agent):
            reward += 0.1

        # Big reward for depositing food at home
        pos = self.agent_positions[agent]
        if np.linalg.norm(pos - self.home_position) < 2.0 and self._is_carrying_food(agent):
            reward += 5.0

        # Collective bonus (shared among all agents when food is deposited)
        if self.collected_resources > 0:
            reward += 0.5

        return reward

    def _get_info(self, agent: str) -> Dict[str, Any]:
        """Get additional information for agent."""
        return {
            "energy": self.agent_energy.get(agent, 0),
            "carrying_food": self._is_carrying_food(agent),
            "episode_metrics": self.episode_metrics.copy()
        }

    def _update_metrics(self):
        """Update episode metrics for analysis."""
        # Calculate average pheromone intensity (information flow)
        if isinstance(self.communication_channel, PheromoneField):
            avg_pheromone = np.mean(self.communication_channel.field)
            self.episode_metrics["information_shared"] = avg_pheromone

        # Resource efficiency
        if self.current_step > 0:
            self.episode_metrics["resource_efficiency"] = self.collected_resources / self.current_step

        # Coordination metric (agents following pheromone trails)
        coordination_score = 0
        for agent in self.agents:
            pheromone = self.communication_channel.receive(agent, self.agent_positions[agent])
            if np.sum(pheromone) > 0.5:  # Following a trail
                coordination_score += 1
        self.episode_metrics["successful_coordination"] = coordination_score / max(1, len(self.agents))


class PredatorPreyEnvironment(EmergenceEnvironment):
    """
    Environment where agents must cooperate to avoid predators.
    Alarm signals and coordinated escape behaviors should emerge.
    """

    def __init__(self, n_agents: int = 8, n_predators: int = 2,
                 predator_speed: float = 1.2, **kwargs):
        self.n_predators = n_predators
        self.predator_speed = predator_speed
        self.predator_positions = {}
        self.predator_targets = {}
        self.alarm_events = 0
        self.successful_escapes = 0

        super().__init__(n_agents, grid_size=(20,20), **kwargs)

    def setup_communication_channel(self):
        """Use symbolic channel for alarm calls."""
        self.communication_channel = SymbolicChannel(
            vocab_size=self.communication_dim if self.communication_dim > 0 else 8,
            max_message_length=2,  # Type + intensity
            broadcast_range=10.0  # Alarm calls travel far
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset with predators."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.current_step = 0

        # Place agents randomly
        for agent in self.agents:
            self.agent_positions[agent] = np.random.rand(2) * np.array(self.grid_size)
            self.agent_orientations[agent] = np.random.rand() * 2 * np.pi
            self.agent_energy[agent] = 100.0

        # Initialize predators
        self.predator_positions = {}
        self.predator_targets = {}
        for i in range(self.n_predators):
            # Start predators at edges
            edge = np.random.choice(4)
            if edge == 0:  # Top
                pos = np.array([np.random.rand() * self.grid_size[0], 0])
            elif edge == 1:  # Right
                pos = np.array([self.grid_size[0] - 1, np.random.rand() * self.grid_size[1]])
            elif edge == 2:  # Bottom
                pos = np.array([np.random.rand() * self.grid_size[0], self.grid_size[1] - 1])
            else:  # Left
                pos = np.array([0, np.random.rand() * self.grid_size[1]])

            self.predator_positions[f"predator_{i}"] = pos
            self.predator_targets[f"predator_{i}"] = None

        self.alarm_events = 0
        self.successful_escapes = 0

        # Reset communication channel
        self.communication_channel.reset()

        # Reset metrics
        self.episode_metrics = {
            "communication_events": 0,
            "successful_coordination": 0,
            "resource_efficiency": 0.0,
            "information_shared": 0.0
        }

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def _update_predators(self):
        """Update predator positions and behaviors."""
        for pred_id, pred_pos in self.predator_positions.items():
            # Find nearest prey
            if self.agents:
                distances = {agent: np.linalg.norm(pred_pos - self.agent_positions[agent])
                           for agent in self.agents}
                target = min(distances, key=distances.get)
                self.predator_targets[pred_id] = target

                # Move toward target
                target_pos = self.agent_positions[target]
                direction = target_pos - pred_pos
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    self.predator_positions[pred_id] = pred_pos + direction * self.predator_speed

                # Catch prey if close enough
                if distances[target] < 1.0:
                    self.agents.remove(target)
                    del self.agent_positions[target]

    def _get_obs(self, agent: str) -> np.ndarray:
        """Observation includes predator detection."""
        if agent not in self.agents:
            return np.zeros(self.observation_space(agent).shape[0])

        pos = self.agent_positions[agent]

        # Detect nearest predator
        predator_info = [0, 0, 0, 0]  # distance, direction_x, direction_y, is_targeted
        if self.predator_positions:
            distances = {pred: np.linalg.norm(pos - pred_pos)
                        for pred, pred_pos in self.predator_positions.items()}
            nearest_pred = min(distances, key=distances.get)
            dist = distances[nearest_pred]

            if dist < 15.0:  # Detection range
                pred_pos = self.predator_positions[nearest_pred]
                direction = (pred_pos - pos) / max(dist, 0.1)
                predator_info = [
                    1.0 - dist / 15.0,  # Normalized inverse distance
                    direction[0],
                    direction[1],
                    1.0 if self.predator_targets.get(nearest_pred) == agent else 0.0
                ]

        # Basic observations
        basic_obs = [
            pos[0] / self.grid_size[0],
            pos[1] / self.grid_size[1],
            np.cos(self.agent_orientations[agent]),
            np.sin(self.agent_orientations[agent]),
            np.clip(self.agent_energy[agent] / 100.0, -1.0, 1.0),  # Group cohesion
            *predator_info,
            0.0  # Padding
        ]

        # Communication observation (alarm calls)
        comm_obs = self.communication_channel.receive(agent, pos)

        return np.concatenate([basic_obs, comm_obs]).astype(np.float32)

    def _get_reward(self, agent: str) -> float:
        """Reward survival and group protection."""
        if agent not in self.agents:
            return 0.0  # Death penalty

        reward = 0.1  # Survival bonus

        # Reward staying with group (safety in numbers)
        pos = self.agent_positions[agent]
        nearby_agents = sum(1 for other in self.agents if other != agent and
                          np.linalg.norm(pos - self.agent_positions[other]) < 5.0)
        reward += nearby_agents * 0.05

        # Penalty for being targeted by predator
        for pred, target in self.predator_targets.items():
            if target == agent:
                reward -= 0.5

        return reward

    def step(self, actions: Dict[str, np.ndarray]):
        """Step with predator updates."""
        try:
            # Update predators first
            self._update_predators()
        except Exception as e:
            print(f"Error updating predators: {e}")

        # Then process agent actions (including potential alarm calls)
        for agent, action in actions.items():
            if agent in self.agents:
                try:
                    # Movement
                    self._process_movement(agent, action[:5])

                    if len(action) > 5:
                        # Check if agent detects predator and sends alarm
                        if self._agent_detects_predator(agent):
                            alarm_signal = action[5:7]
                            if np.max(alarm_signal) > 0.5:  # Strong alarm
                                self.alarm_events += 1
                                self.communication_channel.transmit(
                                    agent, alarm_signal, self.agent_positions[agent]
                                )
                except Exception as e:
                    print(f"Error processing actions for agent {agent}: {e}")
                    # Optionally remove agent or set reward to a penalty
                    if agent in self.agents:
                        self.agents.remove(agent)

        # Call the parent class's step method to handle common logic (rewards, terminations, etc.)
        # This ensures that the environment's state is updated correctly after processing actions.
        observations, rewards, terminations, truncations, infos = super().step(actions)

        return observations, rewards, terminations, truncations, infos

    def _agent_detects_predator(self, agent: str) -> bool:
        """Check if agent can see a predator."""
        pos = self.agent_positions[agent]
        for pred_pos in self.predator_positions.values():
            if np.linalg.norm(pos - pred_pos) < 10.0:
                return True
        return False


    def _get_info(self, agent: str) -> Dict[str, Any]:
        """Get additional info for a single agent."""
        return {
            "energy": self.agent_energy.get(agent, 0),
            "is_targeted": self.predator_targets.get(self._get_nearest_predator_id(agent)) == agent if agent in self.agents else False,
            "alarm_events": self.alarm_events,
            "successful_escapes": self.successful_escapes
        }

    def _get_nearest_predator_id(self, agent: str) -> Optional[str]:
        """Helper to get the ID of the nearest predator."""
        if agent not in self.agents or not self.predator_positions:
            return None
        pos = self.agent_positions[agent]
        distances = {pred: np.linalg.norm(pos - pred_pos)
                     for pred, pred_pos in self.predator_positions.items()}
        return min(distances, key=distances.get)


class TemporalCoordinationEnvironment(EmergenceEnvironment):
    """
    Environment requiring precise temporal coordination.
    Agents must synchronize actions to achieve collective goals.
    """

    def __init__(self, n_agents: int = 6, n_switches: int = 3,
                 sync_window: int = 5, **kwargs):
        self.n_switches = n_switches
        self.sync_window = sync_window
        self.switch_positions = {}
        self.switch_states = {}
        self.activation_history = []
        self.successful_syncs = 0

        super().__init__(n_agents, grid_size=(20,20), **kwargs)

    def setup_communication_channel(self):
        """Continuous channel for timing signals."""
        # Using pheromone field to create temporal gradients
        self.communication_channel = PheromoneField(
            self.grid_size,
            n_chemicals=self.communication_dim if self.communication_dim > 0 else 1,  # Use communication_dim if provided, else default to 1
            decay_rate=0.9,  # Faster decay for temporal precision
            diffusion_rate=0.2
        )

    def _check_synchronization(self) -> bool:
        """Check if all switches were activated within sync window."""
        if len(self.activation_history) < self.n_switches:
            return False

        recent_activations = self.activation_history[-self.n_switches:]
        time_range = max(recent_activations) - min(recent_activations)
        return time_range <= self.sync_window

    def _get_reward(self, agent: str) -> float:
        """Reward based on synchronization success."""
        if agent not in self.agents:
            return 0.0

        reward = -0.01  # Small time penalty

        # Check if agent is near a switch
        pos = self.agent_positions[agent]
        for switch_id, switch_pos in self.switch_positions.items():
            if np.linalg.norm(pos - switch_pos) < 1.5:
                if not self.switch_states[switch_id]:
                    reward += 0.5  # Ready to activate

        # Big reward for successful synchronization
        if self._check_synchronization():
            reward += 10.0
            self.successful_syncs += 1
            # Reset switches
            for switch_id in self.switch_states:
                self.switch_states[switch_id] = False

        return reward


class InformationAsymmetryEnvironment(EmergenceEnvironment):
    """
    Environment where agents have different partial information.
    Communication emerges to share critical knowledge.
    """

    def __init__(self, n_agents: int = 4, n_info_types: int = 3,
                 episode_length: int = 200, render_mode: Optional[str] = None,
                 communication_dim: int = 0, **kwargs):
        self.n_info_types = n_info_types
        self.agent_knowledge = {}  # What each agent knows
        self.hidden_states = {}  # True environment states
        self.information_requests = 0
        self.correct_decisions = 0

        super().__init__(n_agents=n_agents, grid_size=(20,20),
                         episode_length=episode_length, render_mode=render_mode,
                         communication_dim=communication_dim, **kwargs)

    def setup_communication_channel(self):
        """Symbolic channel for information exchange."""
        self.communication_channel = SymbolicChannel(
            vocab_size=self.communication_dim if self.communication_dim > 0 else self.n_info_types * 4,  # Use communication_dim if provided, else default to self.n_info_types * 4
            max_message_length=3,
            broadcast_range=8.0
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset with asymmetric information distribution."""
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.current_step = 0

        # Place agents randomly
        for agent in self.agents:
            self.agent_positions[agent] = np.random.rand(2) * np.array(self.grid_size)
            self.agent_orientations[agent] = np.random.rand() * 2 * np.pi
            self.agent_energy[agent] = 100.0

        # Distribute knowledge asymmetrically
        self.agent_knowledge = {}
        info_types = list(range(self.n_info_types))
        for agent in self.agents:
            # Each agent knows about 1-2 information types
            n_known = np.random.randint(1, 3)
            known_types = np.random.choice(info_types, n_known, replace=False)
            self.agent_knowledge[agent] = set(known_types)

        # Set hidden states
        self.hidden_states = {}
        for i in range(self.n_info_types):
            self.hidden_states[i] = np.random.rand() > 0.5

        self.information_requests = 0
        self.correct_decisions = 0

        # Reset communication channel
        self.communication_channel.reset()

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos

    def _get_obs(self, agent: str) -> np.ndarray:
        """Observation includes partial knowledge."""
        if agent not in self.agents:
            return np.zeros(self.observation_space(agent).shape[0])

        # Encode what this agent knows
        knowledge_vector = np.zeros(self.n_info_types * 2)
        for info_type in self.agent_knowledge.get(agent, set()):
            knowledge_vector[info_type * 2] = 1.0  # Knows about this type
            knowledge_vector[info_type * 2 + 1] = float(self.hidden_states[info_type])

        pos = self.agent_positions[agent]

        # Build basic observation dynamically
        basic_obs_components = [
            pos[0] / self.grid_size[0],
            pos[1] / self.grid_size[1],
            self.agent_energy[agent] / 100.0
        ]

        # Add knowledge vector components up to required size
        knowledge_size = min(len(knowledge_vector), self.base_obs_dim - len(basic_obs_components) - 2) # -2 for time and team size
        basic_obs_components.extend(knowledge_vector[:knowledge_size])

        # Add time progress and team size ratio
        basic_obs_components.append(self.current_step / self.episode_length)
        basic_obs_components.append(len(self.agents) / self.n_agents)

        # Pad if needed
        while len(basic_obs_components) < self.base_obs_dim:
            basic_obs_components.append(0.0)

        basic_obs = np.array(basic_obs_components[:self.base_obs_dim])  # Ensure exact size

        # Communication observation
        comm_obs = self.communication_channel.receive(agent, pos)

        return np.concatenate([basic_obs, comm_obs]).astype(np.float32)

    def _get_reward(self, agent: str) -> float:
        """Reward based on correct decisions and information sharing."""
        if agent not in self.agents:
            return 0.0

        reward = -0.01  # Small time penalty

        # Reward for making correct decisions (simplified: if agent knows true state)
        # In a real scenario, this would involve agent actions based on knowledge
        correct_knowledge_count = 0
        for info_type in self.agent_knowledge.get(agent, set()):
            if self.hidden_states[info_type] == (np.random.rand() > 0.5): # Simplified check
                correct_knowledge_count += 1
        reward += correct_knowledge_count * 0.1

        # Reward for information requests (encourages communication)
        # This would be tied to specific communication actions in a full implementation
        if self.information_requests > 0:
            reward += 0.05 * (self.correct_decisions / max(1, self.information_requests))

        return reward

    def _get_info(self, agent: str) -> Dict[str, Any]:
        """Get additional info for a single agent."""
        return {
            "energy": self.agent_energy.get(agent, 0),
            "known_info_types": list(self.agent_knowledge.get(agent, set())),
            "information_requests": self.information_requests,
            "correct_decisions": self.correct_decisions
        }

    def step(self, actions: Dict[str, np.ndarray]):
        """Execute one environment step for InformationAsymmetryEnvironment."""
        self.current_step += 1

        # Process agent actions
        for agent, action in actions.items():
            if agent in self.agents:
                try:
                    # Movement
                    self._process_movement(agent, action[:5])

                    # Communication (information exchange)
                    if len(action) > 5:
                        comm_signal = action[5:8]  # Assuming 3 communication actions
                        if np.max(comm_signal) > 0.5:  # If there's a significant signal
                            self.communication_channel.transmit(
                                agent, comm_signal, self.agent_positions[agent]
                            )
                            self.episode_metrics["communication_events"] += 1

                    # Simplified decision making (for reward calculation)
                    # In a full implementation, this would be based on agent's action
                    if np.random.rand() < 0.1: # 10% chance to make a decision
                        # Simulate making a decision based on current knowledge + received info
                        # For simplicity, assume a correct decision if agent has full info
                        if len(self.agent_knowledge.get(agent, set())) == self.n_info_types:
                            self.correct_decisions += 1

                except Exception as e:
                    print(f"Error processing actions for agent {agent}: {e}")
                    if agent in self.agents:
                        self.agents.remove(agent)

        # Update communication channel (if applicable, e.g., for decay)
        if isinstance(self.communication_channel, PheromoneField):
            self.communication_channel.update()

        # Calculate rewards
        rewards = {agent: self._get_reward(agent) for agent in self.agents}

        # Check termination
        terminations = {agent: agent not in self.agents for agent in self.possible_agents}
        truncations = {agent: self.current_step >= self.episode_length for agent in self.possible_agents}

        # Update metrics
        self._update_metrics()

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, rewards, terminations, truncations, infos


# Environment Factory
def create_emergence_environment(env_type: str, **kwargs) -> EmergenceEnvironment:
    """Factory function to create different emergence environments."""
    environments = {
        "foraging": ForagingEnvironment,
        "predator_prey": PredatorPreyEnvironment,
        "temporal_coordination": TemporalCoordinationEnvironment,
        "information_asymmetry": InformationAsymmetryEnvironment
    }

    if env_type not in environments:
        raise ValueError(f"Unknown environment type: {env_type}")

    return environments[env_type](**kwargs)


# Example usage showing different emergence scenarios
if __name__ == "__main__":
    print("🌍 Emergence Environments for Communication Research")
    print("=" * 50)

    # Test foraging environment
    print("\n1. Foraging Environment (Ant-inspired)")
    env = create_emergence_environment("foraging", n_agents=5)
    obs, info = env.reset()
    print(f"   Agents: {len(env.agents)}")
    print(f"   Resources: {len(env.resources)}")
    print(f"   Communication: Pheromone field {env.communication_channel.field.shape}")

    # Test predator-prey environment
    print("\n2. Predator-Prey Environment (Alarm signals)")
    env = create_emergence_environment("predator_prey", n_agents=8, n_predators=2)
    obs, info = env.reset()
    print(f"   Prey agents: {len(env.agents)}")
    print(f"   Predators: {len(env.predator_positions)}")
    print(f"   Communication: Symbolic alarm calls")

    # Test temporal coordination
    print("\n3. Temporal Coordination Environment")
    env = create_emergence_environment("temporal_coordination", n_agents=6)
    obs, info = env.reset()
    print(f"   Agents: {len(env.agents)}")
    print(f"   Synchronization window: {env.sync_window} steps")

    # Test information asymmetry
    print("\n4. Information Asymmetry Environment")
    env = create_emergence_environment("information_asymmetry", n_agents=4)
    obs, info = env.reset()
    print(f"   Agents: {len(env.agents)}")
    print(f"   Information types: {env.n_info_types}")
    print(f"   Agent knowledge distribution: {env.agent_knowledge}")

    print("\n✅ All environments created successfully!")