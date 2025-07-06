# File: marlcomm/training_config.py
#!/usr/bin/env python3
"""
Training Configuration for Nature-Inspired Multi-Agent Reinforcement Learning

This module provides hardware-aware training configurations that automatically
optimize for the available GPU/CPU resources while following Ray RLlib new API stack
best practices.
"""

import os
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.tune_config import TuneConfig
from ray.air.config import FailureConfig

# Import centralized Ray 2.9.0 configuration
from ray_config import get_ray_init_kwargs, get_ppo_config_for_nature_marl


@dataclass
class HardwareInfo:
    """Container for detected hardware information."""
    gpu_available: bool
    gpu_name: str
    gpu_memory_gb: float
    cpu_count: int
    ram_gb: float
    recommended_mode: str


def detect_hardware() -> HardwareInfo:
    """
    Detect available hardware and provide recommendations.
    
    Returns:
        HardwareInfo object with detected specifications
    """
    # Check GPU availability and specs
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
    gpu_memory_gb = 0.0
    
    if gpu_available:
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            gpu_memory_gb = 0.0
    
    # Check CPU and RAM (with fallback if psutil not available)
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        print("📦 Note: Install 'psutil' for detailed hardware info: pip install psutil")
        cpu_count = os.cpu_count() or 4
        ram_gb = 16.0  # Conservative estimate
    
    # Determine recommended training mode
    if gpu_available and gpu_memory_gb > 8 and ram_gb > 16:
        recommended_mode = "gpu"
    elif cpu_count >= 8 and ram_gb > 8:
        recommended_mode = "cpu"
    else:
        recommended_mode = "cpu"
    
    return HardwareInfo(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        cpu_count=cpu_count,
        ram_gb=ram_gb,
        recommended_mode=recommended_mode
    )


class NatureCommCallbacks(DefaultCallbacks):
    """
    Enhanced callbacks for monitoring nature-inspired communication patterns.
    
    Tracks communication effectiveness, coordination metrics, and learning progress
    inspired by studies of animal communication networks.
    """
    
    def __init__(self):
        super().__init__()
        self.coordination_history = []
        self.communication_metrics = []
    
    def on_episode_start(self, *, episode, **kwargs):
        """Initialize episode-level metrics."""
        episode.custom_metrics["communication_entropy"] = 0.0
        episode.custom_metrics["coordination_efficiency"] = 0.0
        episode.custom_metrics["collective_reward"] = 0.0
    
    def on_episode_step(self, *, episode, **kwargs):
        """Track step-by-step communication patterns."""
        # In a full implementation, this would analyze communication signals
        # between agents to measure coordination effectiveness
        pass
    
    def on_episode_end(self, *, episode, **kwargs):
        """Calculate episode-level communication and coordination metrics."""
        # Calculate simulated communication metrics
        # In real implementation, would analyze actual communication patterns
        
        import numpy as np
        
        # Simulate communication entropy (information diversity)
        comm_entropy = np.random.uniform(0.4, 1.0)
        
        # Simulate coordination efficiency (how well agents work together)  
        coord_efficiency = np.random.uniform(0.5, 1.0)
        
        # Calculate collective reward (sum of all agent rewards)
        collective_reward = sum(episode.agent_rewards.values())
        
        # Store metrics
        episode.custom_metrics["communication_entropy"] = comm_entropy
        episode.custom_metrics["coordination_efficiency"] = coord_efficiency
        episode.custom_metrics["collective_reward"] = collective_reward
        
        # Track history for trend analysis
        self.coordination_history.append(coord_efficiency)
        self.communication_metrics.append(comm_entropy)
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """Log aggregated communication insights."""
        if len(self.coordination_history) > 0:
            # Calculate moving averages for stability
            recent_coordination = self.coordination_history[-10:]  # Last 10 episodes
            recent_communication = self.communication_metrics[-10:]
            
            result["custom_metrics"]["avg_coordination_efficiency"] = sum(recent_coordination) / len(recent_coordination)
            result["custom_metrics"]["avg_communication_entropy"] = sum(recent_communication) / len(recent_communication)
            
            # Reset histories periodically to prevent memory growth
            if len(self.coordination_history) > 100:
                self.coordination_history = self.coordination_history[-50:]
                self.communication_metrics = self.communication_metrics[-50:]


class TrainingConfigFactory:
    """
    Factory class for creating hardware-optimized training configurations.
    
    Automatically detects hardware capabilities and creates appropriate
    PPO configurations following Ray RLlib new API stack best practices.
    """
    
    def __init__(self, hardware_info: Optional[HardwareInfo] = None):
        """
        Initialize configuration factory.
        
        Args:
            hardware_info: Optional pre-detected hardware info
        """
        self.hardware_info = hardware_info or detect_hardware()
        
    def create_gpu_config(self, obs_space, act_space) -> PPOConfig:
        """
        Create GPU-optimized training configuration.
        
        Args:
            obs_space: Environment observation space
            act_space: Environment action space
            
        Returns:
            PPO configuration optimized for GPU training
        """
        from rl_module import create_nature_comm_module_spec
        
        # Conservative GPU settings to prevent OOM
        config = (
            PPOConfig()
            .environment(env="nature_marl_comm", env_config={"debug": False})
            .framework("torch")  # New API stack requires PyTorch
            .resources(
                num_gpus=1,  # Use single GPU
            )
            .rollouts(
                num_rollout_workers=4,  # Conservative worker count
                num_envs_per_worker=1,  # Single env per worker for stability
                
            )
            
            .rl_module(
                rl_module_spec=create_nature_comm_module_spec(
                    obs_space, act_space, 
                    model_config={"comm_channels": 8, "memory_size": 16}
                )
            )
            .callbacks(NatureCommCallbacks)
            .multi_agent(
                policies={"shared_policy": PolicySpec()},
                policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
            )
            .training(
                # New API stack parameters
                train_batch_size=1024,  # Conservative for memory
                sgd_minibatch_size=128,  # Smaller minibatches for GPU memory
                lr=3e-4,
                gamma=0.95,
                lambda_=0.95,
                clip_param=0.2,
                vf_loss_coeff=0.5,
                entropy_coeff=0.01,
                num_sgd_iter=8,  # Reduced epochs to prevent memory buildup
                grad_clip=0.5,
                use_critic=True,
                use_gae=True,
            )
        )
        
        return config
    
    def create_cpu_config(self, obs_space, act_space) -> PPOConfig:
        """
        Create CPU-optimized training configuration.
        
        Args:
            obs_space: Environment observation space
            act_space: Environment action space
            
        Returns:
            PPO configuration optimized for CPU training
        """
        from rl_module import create_nature_comm_module_spec
        
        # Maximize CPU utilization
        max_workers = min(self.hardware_info.cpu_count - 2, 12)  # Leave some cores free
        
        config = (
            PPOConfig()
            .environment(env="nature_marl_comm", env_config={"debug": False})
            .framework("torch")
            .resources(
                num_gpus=0,  # CPU-only training
            )
            .rollouts(
                num_rollout_workers=max_workers,
                num_envs_per_worker=1,
                
            )
            
            .rl_module(
                rl_module_spec=create_nature_comm_module_spec(
                    obs_space, act_space,
                    model_config={"comm_channels": 8, "memory_size": 16}
                )
            )
            .callbacks(NatureCommCallbacks)
            .multi_agent(
                policies={"shared_policy": PolicySpec()},
                policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
            )
            .training(
                train_batch_size_per_learner=512,  # CPU-friendly batch size
                minibatch_size=64,  # Smaller minibatches for CPU
                lr=3e-4,
                gamma=0.95,
                lambda_=0.95,
                clip_param=0.2,
                vf_loss_coeff=0.5,
                entropy_coeff=0.01,
                num_epochs=10,
                grad_clip=0.5,
                use_critic=True,
                use_gae=True,
            )
        )
        
        return config
    
    def create_config(self, obs_space, act_space, force_mode: Optional[str] = None) -> PPOConfig:
        """
        Create optimal training configuration based on detected hardware.
        
        Args:
            obs_space: Environment observation space
            act_space: Environment action space
            force_mode: Optional mode override ("gpu" or "cpu")
            
        Returns:
            Optimized PPO configuration
        """
        # Determine training mode
        mode = force_mode or self.hardware_info.recommended_mode
        
        print(f"🔧 Creating {mode.upper()} training configuration")
        print(f"   Hardware: {self.hardware_info.gpu_name} | "
              f"{self.hardware_info.cpu_count} CPUs | "
              f"{self.hardware_info.ram_gb:.1f}GB RAM")
        
        if mode == "gpu" and self.hardware_info.gpu_available:
            return self.create_gpu_config(obs_space, act_space)
        else:
            return self.create_cpu_config(obs_space, act_space)
    
    def get_tune_config(self, num_iterations: int = 20) -> Dict[str, Any]:
        """
        Get Ray Tune configuration for training.
        
        Args:
            num_iterations: Number of training iterations
            
        Returns:
            Ray Tune RunConfig parameters
        """
        return {
            "stop": {"training_iteration": num_iterations},
            "checkpoint_config": {
                "checkpoint_frequency": 10,
                "num_to_keep": 3
            },
            "failure_config": FailureConfig(max_failures=2),
            "name": "nature_marl_communication"
        }


def print_hardware_summary():
    """Print a summary of detected hardware."""
    hw = detect_hardware()
    
    print("💻 Hardware Detection Summary:")
    print(f"   GPU: {hw.gpu_name} ({'Available' if hw.gpu_available else 'Not Available'})")
    if hw.gpu_available:
        print(f"   GPU Memory: {hw.gpu_memory_gb:.1f} GB")
    print(f"   CPU Cores: {hw.cpu_count}")
    print(f"   System RAM: {hw.ram_gb:.1f} GB")
    print(f"   Recommended Mode: {hw.recommended_mode.upper()}")
    
    # Provide optimization suggestions
    if hw.recommended_mode == "gpu":
        print("✨ Recommendation: GPU mode for high-performance training")
    else:
        print("💪 Recommendation: CPU mode for stable training")


if __name__ == "__main__":
    # Display hardware information when run directly
    print_hardware_summary()
