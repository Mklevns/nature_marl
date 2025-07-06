# File: marlcomm/reward_engineering_wrapper.py
"""
Unified Reward Engineering Wrapper for MARL Environments

This module provides a robust wrapper that integrates reward engineering
with any PettingZoo environment, tracking communication events and
calculating implicit rewards properly.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
import json

from ray.rllib.env.multi_agent_env import MultiAgentEnv


from reward_engineering import RewardEngineer, RewardContext
from communication_metrics import CommunicationEvent, CommunicationAnalyzer


class RewardEngineeringWrapper(MultiAgentEnv):
    """
    Unified wrapper to integrate reward engineering with environments.
    Properly tracks state, extracts communication, and applies implicit rewards.
    """
    def __init__(self, env, reward_engineer: RewardEngineer,
                 communication_analyzer: Optional[CommunicationAnalyzer] = None):
        """
        Initialize wrapper.

        Args:
            env: Base PettingZoo environment
            reward_engineer: Configured reward engineer
            communication_analyzer: Optional analyzer for metrics
        """
        self.env = env
        self.reward_engineer = reward_engineer
        self.communication_analyzer = communication_analyzer or CommunicationAnalyzer()

        # State tracking
        self.current_step = 0
        self.episode_count = 0
        self.episode_history = []
        self.communication_events = []

        # Cache for efficiency
        self._last_obs = {}
        self._agent_states = defaultdict(lambda: np.zeros(10))  # Default state vector

    def __getattr__(self, name):
        """Delegate to wrapped environment."""
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        """Reset environment and internal state."""
        obs, info = self.env.reset(*args, **kwargs)

        # Reset tracking
        self.current_step = 0
        self.episode_count += 1
        self._last_obs = obs.copy()

        # Clear episode history but keep some for context
        if len(self.episode_history) > 100:
            self.episode_history = self.episode_history[-50:]

        return obs, info

    def step(self, actions: Dict[str, np.ndarray]):
        """
        Step with proper reward engineering and communication tracking.
        """
        # Store pre-step observations
        pre_obs = self._last_obs.copy()

        # Execute environment step
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

        # Extract communication signals from actions
        communication_signals = self._extract_communication_signals(actions)

        # Get true environmental state
        env_state = self._extract_environment_state()

        # Update agent states (could be more sophisticated)
        for agent_id in actions:
            if agent_id in obs:
                self._agent_states[agent_id] = obs[agent_id][:10]  # First 10 dims as state

        # Create proper reward context
        context = RewardContext(
            agent_states=dict(self._agent_states),
            agent_actions=actions,
            agent_observations=pre_obs,  # Use pre-step observations
            communication_signals=communication_signals,
            environmental_state=env_state,
            timestep=self.episode_count * 1000 + self.current_step,  # Global timestep
            episode_history=self.episode_history[-10:]  # Last 10 steps for context
        )

        # Apply reward engineering
        engineered_rewards = {}
        reward_components = {}

        for agent_id in actions:
            if agent_id in rewards:  # Only for active agents
                eng_reward, components = self.reward_engineer.calculate_reward(agent_id, context)
                engineered_rewards[agent_id] = rewards[agent_id] + eng_reward
                reward_components[agent_id] = components

                # Store component breakdown in info
                if agent_id not in infos:
                    infos[agent_id] = {}
                infos[agent_id]['reward_components'] = components
                infos[agent_id]['base_reward'] = rewards[agent_id]
                infos[agent_id]['engineered_reward'] = eng_reward

        # Collect communication events for analysis
        self._collect_communication_events(
            pre_obs, obs, actions, communication_signals, context, infos
        )

        # Add episode history
        self.episode_history.append({
            'step': self.current_step,
            'actions': actions,
            'rewards': engineered_rewards,
            'env_state': env_state
        })

        # Update state
        self.current_step += 1
        self._last_obs = obs.copy()

        # Add metrics to info
        for agent_id in infos:
            infos[agent_id]['communication_sent'] = len(communication_signals.get(agent_id, [])) > 0
            infos[agent_id]['timestep'] = context.timestep

        return obs, engineered_rewards, terminateds, truncateds, infos

    def _extract_communication_signals(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract communication signals from agent actions.
        Handles different action space types and communication channels.
        """
        signals = {}

        for agent_id, action in actions.items():
            if isinstance(action, np.ndarray) and len(action) > 5:
                # Assume communication is in the latter part of action vector
                # Adjust indices based on your environment
                comm_start_idx = 5  # After movement actions
                comm_signal = action[comm_start_idx:]

                # Only count as communication if non-zero
                if np.any(np.abs(comm_signal) > 0.1):
                    signals[agent_id] = comm_signal
            elif hasattr(action, 'communication'):
                # Handle structured action spaces
                signals[agent_id] = action.communication

        return signals

    def _extract_environment_state(self) -> Dict[str, Any]:
        """
        Extract meaningful state from the environment.
        Override this for specific environments.
        """
        state = {
            'step': self.current_step,
            'episode': self.episode_count
        }

        # Try to extract state from wrapped environment
        if hasattr(self.env, 'env'):  # PettingZoo wrapper
            actual_env = self.env.env

            # Common attributes to extract
            attrs_to_check = [
                'collected_resources', 'resources', 'agents', 'agent_positions',
                'predator_positions', 'switch_states', 'hidden_states',
                'food_collected', 'total_food', 'alarm_events'
            ]

            for attr in attrs_to_check:
                if hasattr(actual_env, attr):
                    value = getattr(actual_env, attr)
                    # Convert complex objects to serializable forms
                    if isinstance(value, (list, dict, int, float, bool, str)):
                        state[attr] = value
                    elif isinstance(value, np.ndarray):
                        state[attr] = value.tolist()
                    elif hasattr(value, '__len__'):
                        state[attr] = len(value)

        return state

    def _collect_communication_events(self, pre_obs: Dict, post_obs: Dict,
                                    actions: Dict, signals: Dict,
                                    context: RewardContext, infos: Dict):
        """
        Collect communication events for analysis.
        """
        for sender_id, signal in signals.items():
            # Determine receivers (all other agents within communication range)
            receivers = []

            for agent_id in actions:
                if agent_id != sender_id:
                    # Check if within communication range (environment-specific)
                    if self._is_within_comm_range(sender_id, agent_id):
                        receivers.append(agent_id)

            if receivers:  # Only create event if there are receivers
                # Extract response actions (what receivers did after receiving signal)
                response_actions = {}
                for receiver_id in receivers:
                    if receiver_id in actions:
                        response_actions[receiver_id] = actions[receiver_id]

                event = CommunicationEvent(
                    timestep=context.timestep,
                    sender_id=sender_id,
                    receiver_ids=receivers,
                    message=signal,
                    sender_state=pre_obs.get(sender_id, np.zeros(10)),
                    environmental_context=context.environmental_state,
                    response_actions=response_actions
                )

                self.communication_events.append(event)

                # Update analyzer if available
                if self.communication_analyzer and len(self.communication_events) % 10 == 0:
                    # Analyze recent events periodically
                    self.communication_analyzer.analyze_episode(
                        self.communication_events[-100:]
                    )

    def _is_within_comm_range(self, sender_id: str, receiver_id: str) -> bool:
        """
        Check if two agents are within communication range.
        Override for specific environments.
        """
        # Default: all agents can communicate
        # For spatial environments, check distance
        if hasattr(self, '_agent_positions'):
            sender_pos = self._agent_positions.get(sender_id)
            receiver_pos = self._agent_positions.get(receiver_id)
            if sender_pos is not None and receiver_pos is not None:
                distance = np.linalg.norm(sender_pos - receiver_pos)
                return distance < 10.0  # Communication range

        return True  # Default: global communication

    def get_communication_metrics(self) -> Dict[str, float]:
        """
        Get current communication metrics from analyzer.
        """
        if not self.communication_events:
            return {
                'mutual_information': 0.0,
                'coordination_efficiency': 0.0,
                'communication_frequency': 0.0
            }

        # Get recent analysis
        analysis = self.communication_analyzer.analyze_episode(
            self.communication_events[-500:]
        )

        return analysis['emergence_status']['current_metrics']

    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Get summary of current episode for logging.
        """
        return {
            'episode': self.episode_count,
            'steps': self.current_step,
            'communication_events': len([e for e in self.communication_events
                                       if e.timestep >= self.episode_count * 1000]),
            'reward_distribution': self.reward_engineer.analyze_reward_distribution()
        }

    @property
    def agents(self):
        """
        Provides the 'agents' property, making the wrapper compliant
        with the PettingZoo API.
        """
        # The RLlib wrapper uses `get_agent_ids()` instead of an `agents` property.
        if hasattr(self.env, 'get_agent_ids'):
            return self.env.get_agent_ids()
        # Fallback for other potential wrappers or direct envs.
        if hasattr(self.env, 'agents'):
            return self.env.agents
        # Final fallback to the keys of the last observation dictionary.
        return list(self._last_obs.keys())
