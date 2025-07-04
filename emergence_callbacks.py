# File: marlcomm/emergence_callbacks.py
"""
Real Emergence Tracking Callbacks for RLlib

This module provides callbacks that actually calculate communication metrics
from agent behavior, not simulate them.
"""

from typing import Dict, Any, Optional
import numpy as np
from collections import defaultdict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import Episode

from communication_metrics import CommunicationAnalyzer, CommunicationEvent
from reward_engineering_wrapper import RewardEngineeringWrapper


class RealEmergenceTrackingCallbacks(DefaultCallbacks):
    """
    Callbacks that track real communication emergence metrics.
    """

    def __init__(self):
        super().__init__()
        self.communication_analyzers = {}  # Per-worker analyzers
        self.episode_communication_events = defaultdict(list)
        self.episode_metrics = defaultdict(lambda: defaultdict(float))

    def on_episode_start(self, *, worker, base_env: BaseEnv, policies: Dict[str, Policy],
                        episode: Episode, **kwargs):
        """Initialize episode tracking."""
        episode_id = episode.episode_id

        # Reset episode tracking
        self.episode_communication_events[episode_id] = []
        self.episode_metrics[episode_id] = {
            'communication_frequency': 0,
            'unique_messages': set(),
            'coordination_events': 0,
            'total_steps': 0
        }

        # Initialize custom metrics
        episode.custom_metrics["mutual_information"] = 0.0
        episode.custom_metrics["coordination_efficiency"] = 0.0
        episode.custom_metrics["communication_frequency"] = 0.0
        episode.custom_metrics["protocol_stability"] = 0.0
        episode.custom_metrics["emergence_score"] = 0.0

    def on_episode_step(self, *, worker, base_env: BaseEnv, episode: Episode, **kwargs):
        """Track communication during episode steps."""
        episode_id = episode.episode_id

        # Get the actual environment (unwrap if needed)
        env = base_env.get_sub_environments()[0]
        if isinstance(env, RewardEngineeringWrapper):
            # We have access to proper tracking
            comm_events = env.communication_events

            # Track new events since last step
            new_events = [e for e in comm_events
                         if e not in self.episode_communication_events[episode_id]]
            self.episode_communication_events[episode_id].extend(new_events)

            # Update frequency metric
            if new_events:
                self.episode_metrics[episode_id]['communication_frequency'] += len(new_events)

            # Track unique message patterns
            for event in new_events:
                # Create a hash of the message for uniqueness tracking
                msg_hash = hash(event.message.tobytes())
                self.episode_metrics[episode_id]['unique_messages'].add(msg_hash)

            # Check for coordination (simplified - agents taking similar actions after communication)
            if len(new_events) > 0:
                # Get all agents that received messages
                all_receivers = set()
                for event in new_events:
                    all_receivers.update(event.receiver_ids)

                # Check if receivers took similar actions
                if len(all_receivers) > 1:
                    actions = []
                    for agent_id in all_receivers:
                        if agent_id in event.response_actions:
                            actions.append(event.response_actions[agent_id])

                    if len(actions) > 1:
                        # Simple coordination check: similar actions
                        action_similarity = self._calculate_action_similarity(actions)
                        if action_similarity > 0.7:  # Threshold for coordination
                            self.episode_metrics[episode_id]['coordination_events'] += 1

        else:
            # Fallback: extract from episode data if available
            for agent_id in episode.get_agents():
                agent_info = episode.last_info_for(agent_id)
                if agent_info and 'communication_sent' in agent_info:
                    if agent_info['communication_sent']:
                        self.episode_metrics[episode_id]['communication_frequency'] += 1

        self.episode_metrics[episode_id]['total_steps'] += 1

    def on_episode_end(self, *, worker, base_env: BaseEnv, policies: Dict[str, Policy],
                      episode: Episode, **kwargs):
        """Calculate final episode metrics."""
        episode_id = episode.episode_id
        metrics = self.episode_metrics[episode_id]
        events = self.episode_communication_events[episode_id]

        # Get or create analyzer for this worker
        worker_idx = worker.worker_index if hasattr(worker, 'worker_index') else 0
        if worker_idx not in self.communication_analyzers:
            self.communication_analyzers[worker_idx] = CommunicationAnalyzer()

        analyzer = self.communication_analyzers[worker_idx]

        # Calculate real metrics if we have events
        if events:
            analysis = analyzer.analyze_episode(events)
            emergence_metrics = analysis['emergence_status']['current_metrics']

            # Set real calculated metrics
            episode.custom_metrics["mutual_information"] = emergence_metrics.get('mutual_information', 0.0)
            episode.custom_metrics["coordination_efficiency"] = emergence_metrics.get('coordination_efficiency', 0.0)
            episode.custom_metrics["protocol_stability"] = emergence_metrics.get('protocol_stability', 0.0)
            episode.custom_metrics["emergence_score"] = analysis['emergence_status']['emergence_score']

            # Communication frequency (normalized by steps)
            if metrics['total_steps'] > 0:
                episode.custom_metrics["communication_frequency"] = (
                    metrics['communication_frequency'] / metrics['total_steps']
                )

            # Additional custom metrics
            episode.custom_metrics["unique_message_types"] = len(metrics['unique_messages'])
            episode.custom_metrics["coordination_rate"] = (
                metrics['coordination_events'] / max(1, metrics['communication_frequency'])
            )

        else:
            # No communication events - set minimal metrics
            episode.custom_metrics["mutual_information"] = 0.0
            episode.custom_metrics["coordination_efficiency"] = 0.3  # Baseline
            episode.custom_metrics["communication_frequency"] = 0.0
            episode.custom_metrics["protocol_stability"] = 0.0
            episode.custom_metrics["emergence_score"] = 0.0

        # Clean up episode data
        if episode_id in self.episode_communication_events:
            del self.episode_communication_events[episode_id]
        if episode_id in self.episode_metrics:
            del self.episode_metrics[episode_id]

    def _calculate_action_similarity(self, actions: list) -> float:
        """Calculate similarity between multiple actions."""
        if len(actions) < 2:
            return 0.0

        # Convert to numpy arrays if needed
        actions_array = [np.array(a).flatten() for a in actions]

        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(actions_array)):
            for j in range(i + 1, len(actions_array)):
                # Cosine similarity
                dot_product = np.dot(actions_array[i], actions_array[j])
                norm_product = np.linalg.norm(actions_array[i]) * np.linalg.norm(actions_array[j])
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """Aggregate metrics across episodes."""
        # Add aggregated emergence metrics
        if "env_runners" in result and "custom_metrics" in result["env_runners"]:
            custom_metrics = result["env_runners"]["custom_metrics"]

            # Check for emergence based on mutual information threshold
            mi_mean = custom_metrics.get("mutual_information_mean", 0.0)
            coord_mean = custom_metrics.get("coordination_efficiency_mean", 0.0)

            # Simple emergence detection
            emergence_detected = mi_mean > 0.1 and coord_mean > 0.5

            # Add to results
            result["custom_metrics"]["emergence_detected"] = emergence_detected
            result["custom_metrics"]["emergence_strength"] = custom_metrics.get("emergence_score_mean", 0.0)

            # Log if emergence detected for the first time
            if emergence_detected and not hasattr(self, '_emergence_logged'):
                self._emergence_logged = True
                print(f"\n🎉 Communication emergence detected at iteration {algorithm.iteration}!")
                print(f"   Mutual Information: {mi_mean:.3f}")
                print(f"   Coordination Efficiency: {coord_mean:.3f}")
