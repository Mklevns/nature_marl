# marlcomm/utils/callbacks.py
"""
Fixed callbacks that properly extract data from wrapped environments.
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
import torch

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID

# Import the wrapper to check instance type
from .wrappers import RewardEngineeringWrapper


class BioInspiredCallbacks(DefaultCallbacks):
    """
    Fixed callbacks that properly extract data from wrapped environments.
    """

    def __init__(self):
        super().__init__()
        # Track communication events per episode
        self.episode_communication_events = {}
        self.episode_metrics = {}

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ) -> None:
        """Initialize episode-specific tracking."""
        episode_id = episode.episode_id

        # Initialize tracking dictionaries
        self.episode_communication_events[episode_id] = []
        self.episode_metrics[episode_id] = {
            "message_entropy": [],
            "pheromone_intensity": [],
            "plasticity_magnitude": [],
            "communication_success": [],
            "agent_distances": [],
            "reward_breakdown": {
                "task_reward": [],
                "communication_reward": [],
                "exploration_reward": []
            }
        }

        # Initialize episode user data
        episode.user_data["communication_events"] = []
        episode.user_data["message_entropy"] = []
        episode.user_data["pheromone_intensity"] = []
        episode.user_data["emergent_patterns"] = {
            "message_diversity": [],
            "spatial_clustering": [],
            "temporal_coordination": []
        }

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ) -> None:
        """Extract data from wrapped environment on each step."""
        episode_id = episode.episode_id

        # Get the actual environment (may be wrapped)
        env = base_env.get_sub_environments()[env_index]

        # Extract communication events from wrapper
        if isinstance(env, RewardEngineeringWrapper):
            # Get new events from this step
            new_events = env.get_new_communication_events()
            if new_events:
                self.episode_communication_events[episode_id].extend(new_events)
                episode.user_data["communication_events"].extend(new_events)

            # Get reward breakdown if available
            if hasattr(env, 'get_reward_breakdown'):
                reward_breakdown = env.get_reward_breakdown()
                for reward_type, value in reward_breakdown.items():
                    if reward_type in self.episode_metrics[episode_id]["reward_breakdown"]:
                        self.episode_metrics[episode_id]["reward_breakdown"][reward_type].append(value)

        # Extract bio-inspired metrics from agent infos
        for agent_id in episode.get_agents():
            last_info = episode.last_info_for(agent_id)

            if last_info:
                # Track message entropy
                if "message" in last_info and isinstance(last_info["message"], torch.Tensor):
                    message = last_info["message"]
                    message_probs = torch.softmax(message, dim=-1)
                    entropy = -torch.sum(message_probs * torch.log(message_probs + 1e-8))
                    episode.user_data["message_entropy"].append(entropy.item())

                # Track communication success
                if "communication_success" in last_info:
                    self.episode_metrics[episode_id]["communication_success"].append(
                        last_info["communication_success"]
                    )

                # Track pheromone information
                if "pheromone_intensity" in last_info:
                    episode.user_data["pheromone_intensity"].append(
                        last_info["pheromone_intensity"]
                    )

        # Track emergent patterns (if environment provides this data)
        if hasattr(env, 'get_emergent_metrics'):
            emergent_metrics = env.get_emergent_metrics()
            for pattern_type, value in emergent_metrics.items():
                if pattern_type in episode.user_data["emergent_patterns"]:
                    episode.user_data["emergent_patterns"][pattern_type].append(value)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ) -> None:
        """Compute final metrics for the episode."""
        episode_id = episode.episode_id

        # Communication event statistics
        events = self.episode_communication_events.get(episode_id, [])
        if events:
            episode.custom_metrics["num_communication_events"] = len(events)

            # Analyze communication patterns
            message_types = [e.get("message_type") for e in events if "message_type" in e]
            if message_types:
                unique_types = len(set(message_types))
                episode.custom_metrics["communication_diversity"] = unique_types

        # Message entropy statistics
        if episode.user_data["message_entropy"]:
            episode.custom_metrics["avg_message_entropy"] = np.mean(episode.user_data["message_entropy"])
            episode.custom_metrics["max_message_entropy"] = np.max(episode.user_data["message_entropy"])

        # Pheromone statistics
        if episode.user_data["pheromone_intensity"]:
            episode.custom_metrics["avg_pheromone_intensity"] = np.mean(episode.user_data["pheromone_intensity"])
            episode.custom_metrics["final_pheromone_intensity"] = episode.user_data["pheromone_intensity"][-1]

        # Communication success rate
        success_data = self.episode_metrics[episode_id]["communication_success"]
        if success_data:
            episode.custom_metrics["communication_success_rate"] = np.mean(success_data)

        # Reward breakdown
        for reward_type, values in self.episode_metrics[episode_id]["reward_breakdown"].items():
            if values:
                episode.custom_metrics[f"avg_{reward_type}"] = np.mean(values)
                episode.custom_metrics[f"total_{reward_type}"] = np.sum(values)

        # Emergent pattern metrics
        for pattern_type, values in episode.user_data["emergent_patterns"].items():
            if values:
                episode.custom_metrics[f"emergent_{pattern_type}"] = np.mean(values)

        # Clean up tracking data
        if episode_id in self.episode_communication_events:
            del self.episode_communication_events[episode_id]
        if episode_id in self.episode_metrics:
            del self.episode_metrics[episode_id]

    def on_train_result(
        self,
        *,
        algorithm,
        result: Dict[str, Any],
        **kwargs
    ) -> None:
        """Add high-level bio-inspired metrics to training results."""

        custom_metrics = result.get("custom_metrics", {})

        # Create bio-inspired summary
        bio_metrics = {}

        # Communication efficiency
        if "avg_message_entropy_mean" in custom_metrics:
            # Lower entropy = more structured communication
            bio_metrics["communication_efficiency"] = 1.0 / (1.0 + custom_metrics["avg_message_entropy_mean"])

        # Pheromone utilization
        if "avg_pheromone_intensity_mean" in custom_metrics:
            bio_metrics["pheromone_utilization"] = custom_metrics["avg_pheromone_intensity_mean"]

        # Communication success
        if "communication_success_rate_mean" in custom_metrics:
            bio_metrics["communication_effectiveness"] = custom_metrics["communication_success_rate_mean"]

        # Emergent behavior score
        emergent_scores = []
        for key in custom_metrics:
            if key.startswith("emergent_") and key.endswith("_mean"):
                emergent_scores.append(custom_metrics[key])

        if emergent_scores:
            bio_metrics["emergent_behavior_score"] = np.mean(emergent_scores)

        # Overall bio-inspired score
        if bio_metrics:
            bio_metrics["overall_bio_score"] = np.mean(list(bio_metrics.values()))
            result["bio_inspired_metrics"] = bio_metrics


class RealEmergenceTrackingCallbacks(BioInspiredCallbacks):
    """
    Extended callbacks specifically for tracking emergent communication patterns.
    This version properly extracts data from wrapped environments.
    """

    def __init__(self):
        super().__init__()
        self.protocol_evolution = []
        self.communication_graph = {}

    def on_episode_end(self, **kwargs):
        """Extended analysis of emergent patterns."""
        # First run parent analysis
        super().on_episode_end(**kwargs)

        episode = kwargs["episode"]
        episode_id = episode.episode_id

        # Analyze communication protocol evolution
        if episode_id in self.episode_communication_events:
            events = self.episode_communication_events[episode_id]

            if events:
                # Build communication graph
                for event in events:
                    sender = event.get("sender_id")
                    receiver = event.get("receiver_id")
                    if sender and receiver:
                        key = f"{sender}->{receiver}"
                        self.communication_graph[key] = self.communication_graph.get(key, 0) + 1

                # Analyze protocol structure
                messages = [e.get("message") for e in events if "message" in e]
                if messages and len(messages) > 10:
                    # Simple protocol analysis - check for repeated patterns
                    pattern_count = {}
                    for i in range(len(messages) - 1):
                        if isinstance(messages[i], (list, np.ndarray)):
                            pattern = tuple(np.round(messages[i], 2))  # Discretize for pattern matching
                            pattern_count[pattern] = pattern_count.get(pattern, 0) + 1

                    # Measure protocol convergence
                    if pattern_count:
                        total_patterns = sum(pattern_count.values())
                        most_common = max(pattern_count.values())
                        convergence = most_common / total_patterns
                        episode.custom_metrics["protocol_convergence"] = convergence

                        # Track evolution
                        self.protocol_evolution.append({
                            "episode": episode.episode_id,
                            "convergence": convergence,
                            "unique_patterns": len(pattern_count)
                        })

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Add emergence-specific metrics."""
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)

        # Add communication graph analysis
        if self.communication_graph:
            # Calculate communication network metrics
            num_edges = len(self.communication_graph)
            total_messages = sum(self.communication_graph.values())

            result["emergence_metrics"] = {
                "communication_channels": num_edges,
                "total_messages": total_messages,
                "avg_channel_usage": total_messages / num_edges if num_edges > 0 else 0
            }

            # Add protocol evolution
            if self.protocol_evolution:
                recent_convergence = np.mean([p["convergence"] for p in self.protocol_evolution[-10:]])
                result["emergence_metrics"]["recent_protocol_convergence"] = recent_convergence
