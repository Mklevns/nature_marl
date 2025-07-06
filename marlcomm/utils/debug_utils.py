# File: nature_marl/utils/debug_utils.py
#!/usr/bin/env python3

"""
Debug and analysis utilities for nature-inspired communication systems.
Provides tools for monitoring communication patterns, attention visualization,
and protocol emergence analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import json
from pathlib import Path


class CommunicationAnalyzer:
    """
    Analyzer for understanding emergent communication patterns
    in nature-inspired multi-agent systems.
    """

    def __init__(self, save_dir: str = "./communication_analysis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.communication_data = []
        self.attention_patterns = []
        self.reward_correlations = []

    def analyze_communication_episode(self,
                                    observations: np.ndarray,
                                    pheromone_signals: np.ndarray,
                                    attention_weights: Optional[np.ndarray],
                                    rewards: np.ndarray,
                                    episode_id: int):
        """
        Analyze communication patterns for a single episode.

        Args:
            observations: Agent observations [timesteps, agents, obs_dim]
            pheromone_signals: Pheromone signals [timesteps, agents, pheromone_dim]
            attention_weights: Attention weights [timesteps, agents, agents]
            rewards: Episode rewards [timesteps, agents]
            episode_id: Unique episode identifier
        """
        episode_data = {
            "episode_id": episode_id,
            "communication_diversity": self._calculate_communication_diversity(pheromone_signals),
            "signal_utilization": self._calculate_signal_utilization(pheromone_signals),
            "temporal_consistency": self._calculate_temporal_consistency(pheromone_signals),
            "reward_correlation": self._calculate_reward_correlation(pheromone_signals, rewards)
        }

        if attention_weights is not None:
            episode_data.update({
                "attention_entropy": self._calculate_attention_entropy(attention_weights),
                "communication_networks": self._analyze_communication_networks(attention_weights)
            })

        self.communication_data.append(episode_data)

        # Save raw data for detailed analysis
        np.savez(
            self.save_dir / f"episode_{episode_id}_raw.npz",
            observations=observations,
            pheromone_signals=pheromone_signals,
            attention_weights=attention_weights,
            rewards=rewards
        )

    def _calculate_communication_diversity(self, signals: np.ndarray) -> float:
        """Calculate Shannon entropy of communication signals."""
        # Flatten signals and calculate histogram
        flat_signals = signals.flatten()
        hist, _ = np.histogram(flat_signals, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero entries

        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        return float(entropy)

    def _calculate_signal_utilization(self, signals: np.ndarray) -> float:
        """Calculate what fraction of signal space is actively used."""
        # Calculate variance across pheromone dimensions
        signal_vars = np.var(signals, axis=(0, 1))
        utilization = np.mean(signal_vars > 0.01)  # Threshold for "active" dimensions
        return float(utilization)

    def _calculate_temporal_consistency(self, signals: np.ndarray) -> float:
        """Calculate how consistent signals are over time."""
        if signals.shape[0] < 2:
            return 0.0

        # Calculate correlation between consecutive timesteps
        correlations = []
        for t in range(1, signals.shape[0]):
            corr = np.corrcoef(signals[t-1].flatten(), signals[t].flatten())[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 0.0

    def _calculate_reward_correlation(self, signals: np.ndarray, rewards: np.ndarray) -> float:
        """Calculate correlation between communication and rewards."""
        # Average signals per timestep
        avg_signals = np.mean(np.abs(signals), axis=(1, 2))
        avg_rewards = np.mean(rewards, axis=1)

        if len(avg_signals) > 1 and len(avg_rewards) > 1:
            corr = np.corrcoef(avg_signals, avg_rewards)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        return 0.0

    def _calculate_attention_entropy(self, attention: np.ndarray) -> float:
        """Calculate entropy of attention distributions."""
        # Normalize attention weights
        attention_probs = attention / (np.sum(attention, axis=-1, keepdims=True) + 1e-8)

        # Calculate entropy
        entropy = -np.sum(attention_probs * np.log2(attention_probs + 1e-8), axis=-1)
        return float(np.mean(entropy))

    def _analyze_communication_networks(self, attention: np.ndarray) -> Dict[str, float]:
        """Analyze communication network properties."""
        # Average attention over time
        avg_attention = np.mean(attention, axis=0)

        # Network density (how connected the agents are)
        threshold = 0.1  # Minimum attention threshold
        connections = (avg_attention > threshold).astype(int)
        density = np.sum(connections) / (connections.shape[0] * connections.shape[1])

        # Centralization (how much communication flows through central agents)
        out_degrees = np.sum(connections, axis=1)
        max_centralization = np.max(out_degrees) / (connections.shape[0] - 1)

        return {
            "network_density": float(density),
            "centralization": float(max_centralization),
            "avg_attention_strength": float(np.mean(avg_attention))
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive communication analysis report."""
        if not self.communication_data:
            return {"error": "No communication data available"}

        # Aggregate statistics
        diversity_trend = [ep["communication_diversity"] for ep in self.communication_data]
        utilization_trend = [ep["signal_utilization"] for ep in self.communication_data]
        consistency_trend = [ep["temporal_consistency"] for ep in self.communication_data]
        reward_correlation_trend = [ep["reward_correlation"] for ep in self.communication_data]

        report = {
            "total_episodes_analyzed": len(self.communication_data),
            "communication_evolution": {
                "diversity_trend": diversity_trend,
                "final_diversity": diversity_trend[-1] if diversity_trend else 0,
                "diversity_improvement": diversity_trend[-1] - diversity_trend[0] if len(diversity_trend) > 1 else 0
            },
            "signal_efficiency": {
                "utilization_trend": utilization_trend,
                "final_utilization": utilization_trend[-1] if utilization_trend else 0,
                "avg_utilization": np.mean(utilization_trend) if utilization_trend else 0
            },
            "protocol_stability": {
                "consistency_trend": consistency_trend,
                "avg_consistency": np.mean(consistency_trend) if consistency_trend else 0
            },
            "reward_alignment": {
                "correlation_trend": reward_correlation_trend,
                "avg_correlation": np.mean(reward_correlation_trend) if reward_correlation_trend else 0
            }
        }

        # Add attention analysis if available
        attention_episodes = [ep for ep in self.communication_data if "attention_entropy" in ep]
        if attention_episodes:
            attention_entropies = [ep["attention_entropy"] for ep in attention_episodes]
            report["attention_analysis"] = {
                "entropy_trend": attention_entropies,
                "avg_entropy": np.mean(attention_entropies),
                "final_entropy": attention_entropies[-1]
            }

        # Save report
        with open(self.save_dir / "communication_report.json", "w") as f:
            json.dump(report, f, indent=2)

        return report

    def visualize_communication_evolution(self, save_plots: bool = True):
        """Create visualizations of communication evolution."""
        if not self.communication_data:
            print("No communication data to visualize")
            return

        episodes = [ep["episode_id"] for ep in self.communication_data]
        diversity = [ep["communication_diversity"] for ep in self.communication_data]
        utilization = [ep["signal_utilization"] for ep in self.communication_data]
        consistency = [ep["temporal_consistency"] for ep in self.communication_data]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Communication diversity over time
        axes[0, 0].plot(episodes, diversity, 'b-', linewidth=2)
        axes[0, 0].set_title('Communication Diversity Evolution')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Shannon Entropy')
        axes[0, 0].grid(True, alpha=0.3)

        # Signal utilization over time
        axes[0, 1].plot(episodes, utilization, 'g-', linewidth=2)
        axes[0, 1].set_title('Signal Space Utilization')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Fraction of Dimensions Used')
        axes[0, 1].grid(True, alpha=0.3)

        # Temporal consistency
        axes[1, 0].plot(episodes, consistency, 'r-', linewidth=2)
        axes[1, 0].set_title('Temporal Consistency')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Correlation Coefficient')
        axes[1, 0].grid(True, alpha=0.3)

        # Communication-reward correlation
        reward_corr = [ep["reward_correlation"] for ep in self.communication_data]
        axes[1, 1].plot(episodes, reward_corr, 'm-', linewidth=2)
        axes[1, 1].set_title('Communication-Reward Correlation')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Correlation Coefficient')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            plt.savefig(self.save_dir / "communication_evolution.png", dpi=300, bbox_inches='tight')

        plt.show()


class CommunicationCallback(DefaultCallbacks):
    """
    RLlib callback for monitoring communication during training.
    """

    def __init__(self):
        super().__init__()
        self.analyzer = CommunicationAnalyzer()
        self.episode_data = {}

    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        """Collect communication data during episode steps."""
        # Extract communication information from the last step
        for agent_id in episode.get_agents():
            if agent_id not in self.episode_data:
                self.episode_data[agent_id] = {
                    "observations": [],
                    "pheromone_signals": [],
                    "attention_weights": [],
                    "rewards": []
                }

            # Get latest information from the episode
            last_info = episode.last_info_for(agent_id)
            if last_info and "communication_stats" in last_info:
                self.episode_data[agent_id]["pheromone_signals"].append(
                    last_info["pheromone_signals"]
                )

            # Store observations and rewards
            self.episode_data[agent_id]["observations"].append(
                episode.last_observation_for(agent_id)
            )
            self.episode_data[agent_id]["rewards"].append(
                episode.last_reward_for(agent_id)
            )

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        """Analyze episode communication data."""
        if self.episode_data:
            # Convert to numpy arrays and analyze
            agent_ids = list(self.episode_data.keys())

            # Stack data from all agents
            observations = np.stack([
                np.array(self.episode_data[aid]["observations"])
                for aid in agent_ids
            ], axis=1)

            rewards = np.stack([
                np.array(self.episode_data[aid]["rewards"])
                for aid in agent_ids
            ], axis=1)

            # Only analyze if we have pheromone data
            if self.episode_data[agent_ids[0]]["pheromone_signals"]:
                pheromone_signals = np.stack([
                    np.array(self.episode_data[aid]["pheromone_signals"])
                    for aid in agent_ids
                ], axis=1)

                attention_weights = None  # Would need to be extracted from model

                self.analyzer.analyze_communication_episode(
                    observations=observations,
                    pheromone_signals=pheromone_signals,
                    attention_weights=attention_weights,
                    rewards=rewards,
                    episode_id=episode.episode_id
                )

        # Reset episode data
        self.episode_data = {}

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Generate periodic communication reports."""
        if result["training_iteration"] % 10 == 0:  # Every 10 iterations
            report = self.analyzer.generate_report()
            print(f"\n--- Communication Analysis (Iteration {result['training_iteration']}) ---")
            print(f"Episodes analyzed: {report.get('total_episodes_analyzed', 0)}")

            if "communication_evolution" in report:
                print(f"Current diversity: {report['communication_evolution']['final_diversity']:.3f}")
                print(f"Signal utilization: {report['signal_efficiency']['final_utilization']:.3f}")
                print(f"Protocol stability: {report['protocol_stability']['avg_consistency']:.3f}")

            # Add metrics to result for TensorBoard logging
            result.update({
                "custom_metrics/communication_diversity": report.get("communication_evolution", {}).get("final_diversity", 0),
                "custom_metrics/signal_utilization": report.get("signal_efficiency", {}).get("final_utilization", 0),
                "custom_metrics/protocol_stability": report.get("protocol_stability", {}).get("avg_consistency", 0),
                "custom_metrics/reward_correlation": report.get("reward_alignment", {}).get("avg_correlation", 0)
            })


# Utility functions for analysis
def load_and_analyze_checkpoint(checkpoint_path: str, env_name: str) -> Dict[str, Any]:
    """
    Load a trained model and analyze its communication patterns.

    Args:
        checkpoint_path: Path to the model checkpoint
        env_name: Name of the registered environment

    Returns:
        Analysis results dictionary
    """
    # Restore algorithm
    algo = Algorithm.from_checkpoint(checkpoint_path)
    analyzer = CommunicationAnalyzer()

    # Run evaluation episodes
    env = algo.workers.local_worker().env
    num_episodes = 10

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0

        episode_obs = []
        episode_signals = []
        episode_rewards = []

        while not done and step < 500:
            actions = {}
            for agent_id in obs.keys():
                action = algo.compute_single_action(obs[agent_id], policy_id="shared_policy")
                actions[agent_id] = action

            obs, rewards, dones, truncated, infos = env.step(actions)

            # Collect data
            episode_obs.append(list(obs.values()))
            episode_rewards.append(list(rewards.values()))

            # Extract communication signals if available
            if infos and any("pheromone_signals" in info for info in infos.values()):
                signals = [info.get("pheromone_signals", []) for info in infos.values()]
                episode_signals.append(signals)

            done = all(dones.values()) or all(truncated.values())
            step += 1

        # Analyze episode if we have signal data
        if episode_signals:
            analyzer.analyze_communication_episode(
                observations=np.array(episode_obs),
                pheromone_signals=np.array(episode_signals),
                attention_weights=None,
                rewards=np.array(episode_rewards),
                episode_id=episode
            )

    return analyzer.generate_report()
