# File: nature_marl/real_emergence_callbacks.py
"""
Definitive Callbacks for Emergence and Performance Tracking in RLlib

This module provides the primary callback class, RealEmergenceTrackingCallbacks,
which integrates two key functions:
1.  **Emergence Analysis**: Calculates sophisticated communication metrics by
    analyzing CommunicationEvent objects produced by the environment wrapper.
2.  **Performance Monitoring**: Tracks hardware utilization (GPU/CPU) to
    correlate system performance with training progress.

This class is intended to be the single source of callbacks for all training runs.
"""

from typing import Dict, Any, List
import numpy as np
from collections import defaultdict
import torch

# Optional imports for hardware monitoring; the code will not fail if they are not present.
try:
    import GPUtil
except ImportError:
    GPUtil = None
try:
    import psutil
except ImportError:
    psutil = None

# Ray and RLlib Core Imports
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import Episode

# Local MARL-DevGPT Framework Imports
from communication_metrics import CommunicationAnalyzer, CommunicationEvent
from reward_engineering_wrapper import RewardEngineeringWrapper


class RealEmergenceTrackingCallbacks(DefaultCallbacks):
    """
    Consolidated callbacks to track both communication emergence and hardware performance.

    This class interfaces with the RewardEngineeringWrapper to get detailed
    communication events and uses system utilities to log hardware stats,
    providing a holistic view of the training process.
    """

    def __init__(self):
        super().__init__()
        # Each worker needs its own analyzer instance to avoid state collision
        self.worker_analyzers: Dict[int, CommunicationAnalyzer] = {}
        # Store raw communication events per episode within a worker's context
        self.episode_raw_events: Dict[str, List[CommunicationEvent]] = defaultdict(list)

    def _get_analyzer(self, worker_id: int) -> CommunicationAnalyzer:
        """Lazily creates and retrieves a CommunicationAnalyzer for a given worker."""
        if worker_id not in self.worker_analyzers:
            self.worker_analyzers[worker_id] = CommunicationAnalyzer()
        return self.worker_analyzers[worker_id]

    def on_episode_start(self, *, worker, base_env: BaseEnv, episode: Episode, **kwargs):
        """Initialize episode-specific data stores and metrics."""
        episode_id = episode.episode_id
        self.episode_raw_events[episode_id] = []

        # Initialize all custom metrics that will be reported to Ray Tune.
        # This ensures they appear in results even if the episode has no communication.
        episode.custom_metrics["mutual_information"] = 0.0
        episode.custom_metrics["coordination_efficiency"] = 0.0
        episode.custom_metrics["communication_frequency"] = 0.0
        episode.custom_metrics["protocol_stability"] = 0.0
        episode.custom_metrics["emergence_score"] = 0.0
        episode.custom_metrics["unique_message_types"] = 0
        episode.custom_metrics["semantic_coherence"] = 0.0
        episode.custom_metrics["channel_capacity"] = 0.0

    def on_episode_step(self, *, worker, base_env: BaseEnv, episode: Episode, **kwargs):
        """
        At each step, retrieve newly generated communication events from the environment wrapper.
        """
        episode_id = episode.episode_id
        # Assumes the environment is wrapped with RewardEngineeringWrapper
        env = base_env.get_sub_environments()[0]
        if isinstance(env, RewardEngineeringWrapper):
            # The wrapper is responsible for capturing events during its `step` method
            new_events_this_step = env.get_new_communication_events()
            self.episode_raw_events[episode_id].extend(new_events_this_step)

    def on_episode_end(self, *, worker, base_env: BaseEnv, episode: Episode, **kwargs):
        """
        At the end of an episode, analyze all collected communication events
        and populate the episode's custom metrics for logging.
        """
        episode_id = episode.episode_id
        worker_idx = worker.worker_index if hasattr(worker, 'worker_index') else 0

        analyzer = self._get_analyzer(worker_idx)
        all_events_for_episode = self.episode_raw_events.get(episode_id, [])

        if all_events_for_episode:
            # Perform a full analysis on the collected events for this episode
            analysis = analyzer.analyze_episode(all_events_for_episode)
            emergence_metrics = analysis['emergence_status']['current_metrics']
            semantic_metrics = analysis['semantic_analysis']

            # Populate custom metrics for this episode, which RLlib will average
            episode.custom_metrics["mutual_information"] = emergence_metrics.get('mutual_information', 0.0)
            episode.custom_metrics["coordination_efficiency"] = emergence_metrics.get('coordination_efficiency', 0.0)
            episode.custom_metrics["protocol_stability"] = emergence_metrics.get('protocol_stability', 0.0)
            episode.custom_metrics["emergence_score"] = analysis['emergence_status']['emergence_score']
            episode.custom_metrics["channel_capacity"] = emergence_metrics.get('channel_capacity', 0.0)
            episode.custom_metrics["semantic_coherence"] = semantic_metrics.get('semantic_coherence', 0.0)
            episode.custom_metrics["unique_message_types"] = semantic_metrics.get('message_clusters', 0)

            # Normalize communication frequency by the number of agents and steps
            num_agents = len(episode.agent_rewards)
            total_agent_steps = episode.length * max(1, num_agents)
            episode.custom_metrics["communication_frequency"] = len(all_events_for_episode) / max(1, total_agent_steps)

        # Clean up episode-specific data to prevent memory leaks
        if episode_id in self.episode_raw_events:
            del self.episode_raw_events[episode_id]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """
        Called at the end of a training iteration. This method adds aggregated
        emergence status and hardware performance metrics to the final result dict.
        """
        # --- Part 1: Global Emergence Detection ---
        # RLlib automatically averages the per-episode custom metrics.
        # We can access these via `result["custom_metrics"]`.
        if "custom_metrics" in result:
            emergence_score_mean = result["custom_metrics"].get("emergence_score_mean", 0.0)

            # Define a threshold for declaring "global" emergence
            if emergence_score_mean > 0.6 and not hasattr(self, '_global_emergence_logged'):
                self._global_emergence_logged = True  # Log only once
                print(f"\n" + "="*80)
                print(f"🎉 GLOBAL COMMUNICATION EMERGENCE DETECTED at training iteration {algorithm.iteration}!")
                print(f"   Mean Emergence Score: {emergence_score_mean:.3f}")
                print(f"   Mean Mutual Information: {result['custom_metrics'].get('mutual_information_mean', 0.0):.3f}")
                print(f"   Mean Coordination Efficiency: {result['custom_metrics'].get('coordination_efficiency_mean', 0.0):.3f}")
                print(f"="*80 + "\n")

        # --- Part 2: Hardware Performance Monitoring ---
        # Add GPU metrics if available
        if torch.cuda.is_available():
            result["custom_metrics"]["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / 1e6
            result["custom_metrics"]["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / 1e6
            if GPUtil:
                gpus = GPUtil.getGPUs()
                if gpus:
                    result["custom_metrics"]["gpu_utilization_percent"] = gpus[0].load * 100
                    result["custom_metrics"]["gpu_temperature_celsius"] = gpus[0].temperature

        # Add CPU and RAM metrics if available
        if psutil:
            result["custom_metrics"]["cpu_percent"] = psutil.cpu_percent(interval=None)
            result["custom_metrics"]["memory_percent"] = psutil.virtual_memory().percent
