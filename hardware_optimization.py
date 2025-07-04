# File: marlcomm/hardware_optimization.py
"""
Hardware Optimization for High-Performance MARL Training

This module optimizes the framework for high-end hardware like:
- RTX 4070 GPU (12GB VRAM)
- Ryzen 9 3900X (12 cores, 24 threads)
- 64GB RAM

Maximizes throughput and reduces training time significantly.
"""

import torch
import psutil
import GPUtil
import os
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import numpy as np


@dataclass
class HardwareProfile:
    """Detected hardware configuration."""
    # GPU
    gpu_available: bool
    gpu_name: str
    gpu_memory_gb: float
    gpu_compute_capability: Tuple[int, int]

    # CPU
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_name: str

    # Memory
    ram_total_gb: float
    ram_available_gb: float

    # Optimal settings
    optimal_batch_size: int
    optimal_num_workers: int
    optimal_num_envs_per_worker: int
    optimal_gpu_memory_fraction: float


class HardwareOptimizer:
    """Optimizes training configuration for available hardware."""

    def __init__(self):
        self.profile = self._detect_hardware()
        self._print_hardware_summary()

    def _detect_hardware(self) -> HardwareProfile:
        """Detect and profile available hardware."""
        # GPU Detection
        gpu_available = torch.cuda.is_available()
        gpu_name = "None"
        gpu_memory_gb = 0.0
        gpu_compute = (0, 0)

        if gpu_available:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_name = gpu.name
                gpu_memory_gb = gpu.memoryTotal / 1024  # Convert MB to GB
            else:
                # Fallback to torch
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            gpu_compute = torch.cuda.get_device_capability(0)

        # CPU Detection
        cpu_cores_physical = psutil.cpu_count(logical=False) or 1
        cpu_cores_logical = psutil.cpu_count(logical=True) or 1

        # Try to get CPU name
        try:
            import platform
            cpu_name = platform.processor()
        except:
            cpu_name = f"{cpu_cores_physical} core CPU"

        # Memory Detection
        mem = psutil.virtual_memory()
        ram_total_gb = mem.total / (1024**3)
        ram_available_gb = mem.available / (1024**3)

        # Calculate optimal settings
        profile = HardwareProfile(
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            gpu_compute_capability=gpu_compute,
            cpu_cores_physical=cpu_cores_physical,
            cpu_cores_logical=cpu_cores_logical,
            cpu_name=cpu_name,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            optimal_batch_size=0,  # Will calculate
            optimal_num_workers=0,  # Will calculate
            optimal_num_envs_per_worker=0,  # Will calculate
            optimal_gpu_memory_fraction=0.9  # Use 90% of GPU memory
        )

        # Calculate optimal settings based on hardware
        self._calculate_optimal_settings(profile)

        return profile

    def _calculate_optimal_settings(self, profile: HardwareProfile):
        """Calculate optimal training settings for detected hardware."""

        # For RTX 4070 (12GB) + Ryzen 9 3900X (12 cores) + 64GB RAM
        if profile.gpu_memory_gb >= 10 and profile.cpu_cores_physical >= 12 and profile.ram_total_gb >= 32:
            # High-end configuration
            profile.optimal_batch_size = 8192  # Large batch for GPU efficiency
            profile.optimal_num_workers = 10   # Leave 2 cores for system/learner
            profile.optimal_num_envs_per_worker = 4  # 40 parallel environments total

        elif profile.gpu_memory_gb >= 8:
            # Mid-range GPU configuration
            profile.optimal_batch_size = 4096
            profile.optimal_num_workers = min(profile.cpu_cores_physical - 2, 8)
            profile.optimal_num_envs_per_worker = 2

        elif profile.gpu_available:
            # Low-end GPU configuration
            profile.optimal_batch_size = 2048
            profile.optimal_num_workers = min(profile.cpu_cores_physical - 2, 6)
            profile.optimal_num_envs_per_worker = 1

        else:
            # CPU-only configuration
            profile.optimal_batch_size = 1024
            profile.optimal_num_workers = min(profile.cpu_cores_physical - 1, 12)
            profile.optimal_num_envs_per_worker = 1

    def _print_hardware_summary(self):
        """Print detected hardware configuration."""
        print("🖥️  Hardware Configuration Detected")
        print("=" * 60)

        print(f"GPU: {self.profile.gpu_name}")
        if self.profile.gpu_available:
            print(f"  Memory: {self.profile.gpu_memory_gb:.1f} GB")
            print(f"  Compute Capability: {self.profile.gpu_compute_capability}")

        print(f"\nCPU: {self.profile.cpu_name}")
        print(f"  Physical Cores: {self.profile.cpu_cores_physical}")
        print(f"  Logical Cores: {self.profile.cpu_cores_logical}")

        print(f"\nRAM: {self.profile.ram_total_gb:.1f} GB total")
        print(f"  Available: {self.profile.ram_available_gb:.1f} GB")

        print(f"\n⚡ Optimal Settings:")
        print(f"  Batch Size: {self.profile.optimal_batch_size}")
        print(f"  Workers: {self.profile.optimal_num_workers}")
        print(f"  Envs per Worker: {self.profile.optimal_num_envs_per_worker}")
        print(f"  Total Parallel Envs: {self.profile.optimal_num_workers * self.profile.optimal_num_envs_per_worker}")
        print("=" * 60)

    def get_optimized_ppo_config(self,
                                env: str,
                                env_config: Dict[str, Any],
                                rl_module_spec: Any) -> PPOConfig:
        """
        Create PPO configuration optimized for detected hardware.

        This configuration maximizes throughput on high-end systems.

        Args:
            env: Environment name
            env_config: Environment configuration dict
            rl_module_spec: RL module specification

        Returns:
            PPOConfig: Fully configured PPO algorithm config
        """
        config = PPOConfig()
        config = config.experimental(_enable_new_api_stack=True)

        # Environment configuration
        config = config.environment(
            env=env,
            env_config=env_config,
            disable_env_checking=True  # Skip checks for speed
        )

        # Framework
        config = config.framework(
            framework="torch",
        )

        # Resources - optimize for your hardware
        config = config.resources(
            num_gpus=1 if self.profile.gpu_available else 0,
        )

        # Rollout workers - maximize environment throughput
        config = config.rollouts(
            num_rollout_workers=self.profile.optimal_num_workers,
            num_envs_per_worker=self.profile.optimal_num_envs_per_worker,
            rollout_fragment_length=512,  # Steps per worker before sending to learner
            batch_mode="complete_episodes",
        )

        # Training configuration - large batches for GPU
        # FIXED: Ensure sgd_minibatch_size <= train_batch_size
        train_batch_size = self.profile.optimal_batch_size

        # Calculate appropriate minibatch size
        if self.profile.gpu_available:
            # For GPU: use larger minibatches but ensure they fit
            sgd_minibatch_size = min(256, train_batch_size // 4)
        else:
            # For CPU: smaller minibatches
            sgd_minibatch_size = min(128, train_batch_size // 8)

        # Ensure minibatch size is at least 32 and divides train_batch_size evenly
        sgd_minibatch_size = max(32, sgd_minibatch_size)

        # Make sure train_batch_size is divisible by sgd_minibatch_size
        if train_batch_size % sgd_minibatch_size != 0:
            # Adjust minibatch size to be a divisor of train_batch_size
            for size in [256, 128, 64, 32]:
                if size <= train_batch_size and train_batch_size % size == 0:
                    sgd_minibatch_size = size
                    break
            else:
                # If no good divisor found, make train_batch_size a multiple of sgd_minibatch_size
                train_batch_size = ((train_batch_size // sgd_minibatch_size) + 1) * sgd_minibatch_size

        # Configure training with minimal PPO parameters
        # Only use the most essential parameters that are guaranteed to work
        # File: marlcomm/hardware_optimization.py
        # Configure training with minimal PPO parameters
        # Only use the most essential parameters that are guaranteed to work
        config = config.training(
            train_batch_size=train_batch_size,
            sgd_minibatch_size=sgd_minibatch_size,
            num_sgd_iter=15 if self.profile.gpu_available else 10,
            lr=3e-4,
            gamma=0.99,
            model={
                "fcnet_hiddens": [256, 256] if self.profile.gpu_available else [128, 128],
                "fcnet_activation": "relu",
            }
        )
        # Multi-agent setup for a shared policy
        config = config.multi_agent(
            policies={"shared_policy": (None, rl_module_spec.module_specs["agent_0"].observation_space, rl_module_spec.module_specs["agent_0"].action_space, {})},
            policy_mapping_fn=(
                lambda agent_id, episode, worker, **kwargs: "shared_policy"
            )
        )

        # Fault tolerance for long runs
        config = config.fault_tolerance(
            recreate_failed_workers=True,
        )

        # Reporting
        config = config.reporting(
            keep_per_episode_custom_metrics=True,
            metrics_num_episodes_for_smoothing=100,
        )

        # Explicitly return the configured PPOConfig object
        return config

    def optimize_ray_init(self) -> Dict[str, Any]:
        """
        Get optimal Ray initialization parameters.

        Returns configuration for ray.init()
        """
        # Calculate memory allocation
        # Reserve some RAM for system (8GB) and environments
        object_store_memory = int(20 * 1e9)  # Set to 20GB to fit within /dev/shm

        # CPU allocation
        # Leave 1 core for system, 1 for main thread
        num_cpus = self.profile.cpu_cores_logical - 2

        ray_config = {
            "num_cpus": num_cpus,
            "num_gpus": 1 if self.profile.gpu_available else 0,
            "object_store_memory": object_store_memory,

            # Dashboard for monitoring
            "include_dashboard": False,
            "dashboard_host": "0.0.0.0",

            # Optimizations
            "_system_config": {
                # Faster object transfers
                "object_spilling_config": json.dumps({
                    "type": "filesystem",
                    "params": {
                        "directory_path": "/tmp/ray_spill",  # Use fast SSD
                    }
                }),

                # Scheduling optimizations
                "scheduler_spread_threshold": 0.5,

                # Larger object sizes for batch efficiency
                "max_direct_call_object_size": 1000 * 1024 * 1024,  # 1GB

                # Plasma store optimizations

            },

            # Logging
            "logging_level": "WARNING",  # Reduce logging overhead

            # Local mode OFF for production
            "local_mode": False,
        }

        return ray_config

    def get_environment_optimizations(self) -> Dict[str, Any]:
        """Get environment-specific optimizations."""
        return {
            # Vectorization settings
            "vectorize_environments": True,
            "num_envs": self.profile.optimal_num_workers * self.profile.optimal_num_envs_per_worker,

            # Environment step optimizations
            "batch_mode": "complete_episodes",
            "compress_observations": True,  # Compress large observations

            # Memory optimizations
            "zero_copy": True,  # Avoid copying data between processes

            # GPU environment settings (if applicable)
            "gpu_environments": self.profile.gpu_available,
        }

    def optimize_model_architecture(self,
                                  observation_dim: int,
                                  action_dim: int,
                                  communication_dim: int) -> Dict[str, Any]:
        """
        Optimize neural network architecture for hardware.

        Larger models for GPUs, efficient models for CPUs.
        """
        if self.profile.gpu_available and self.profile.gpu_memory_gb >= 10:
            # Large model for RTX 4070
            return {
                "encoder_hiddens": [512, 512, 256],
                "communication_hiddens": [256, 128],
                "policy_hiddens": [256, 256],
                "value_hiddens": [256, 256],
                "activation": "gelu",  # Better for deep networks
                "use_attention": True,  # Self-attention for agent interactions
                "attention_dim": 128,
                "dropout": 0.1,
                "batch_norm": True,
            }
        elif self.profile.gpu_available:
            # Medium model for smaller GPUs
            return {
                "encoder_hiddens": [256, 256],
                "communication_hiddens": [128, 64],
                "policy_hiddens": [128, 128],
                "value_hiddens": [128, 128],
                "activation": "relu",
                "use_attention": False,
                "dropout": 0.0,
                "batch_norm": False,
            }
        else:
            # Efficient model for CPU
            return {
                "encoder_hiddens": [128, 128],
                "communication_hiddens": [64, 32],
                "policy_hiddens": [64, 64],
                "value_hiddens": [64, 64],
                "activation": "relu",
                "use_attention": False,
                "dropout": 0.0,
                "batch_norm": False,
            }


class OptimizedTrainingPipeline:
    """High-performance training pipeline using all optimizations."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.optimizer = HardwareOptimizer()

        # Set environment variables for performance
        self._set_performance_env_vars()

    def _set_performance_env_vars(self):
        """Set environment variables for maximum performance."""
        # PyTorch optimizations
        os.environ["OMP_NUM_THREADS"] = str(self.optimizer.profile.cpu_cores_physical)
        os.environ["MKL_NUM_THREADS"] = str(self.optimizer.profile.cpu_cores_physical)

        # GPU optimizations
        if self.optimizer.profile.gpu_available:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async CUDA
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

            # Enable TF32 on Ampere GPUs (RTX 30/40 series)
            if self.optimizer.profile.gpu_compute_capability >= (8, 0):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Ray optimizations
        os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # Disable memory monitor
        os.environ["RAY_event_stats"] = "0"  # Disable event stats

    def create_optimized_training_config(self,
                                       env_name: str,
                                       env_config: Dict[str, Any],
                                       rl_module_spec: Any) -> Dict[str, Any]:
        """Create complete optimized training configuration."""

        # Get optimized PPO config
        ppo_config = self.optimizer.get_optimized_ppo_config(
            env_name, env_config, rl_module_spec
        )

        # Add custom metrics for monitoring
        ppo_config = ppo_config.callbacks(OptimizedCallbacks)

        # Training configuration
        train_config = {
            "algorithm": ppo_config,
            "stop_criteria": {
                "training_iteration": 1000,
                "env_runners/episode_reward_mean": 500,
                "time_total_s": 3600 * 4,  # 4 hour maximum
            },
            "checkpoint_config": {
                "checkpoint_frequency": 50,
                "checkpoint_at_end": True,
                "num_to_keep": 3,
            },
            "progress_reporter": "json",  # Efficient progress reporting
            "verbose": 1,
        }

        return train_config

    def run_optimized_training(self, train_config: Dict[str, Any]):
        """Run training with all optimizations enabled."""

        print(f"\n🚀 Starting optimized training: {self.experiment_name}")
        print(f"   Total parallel environments: "
              f"{self.optimizer.profile.optimal_num_workers * self.optimizer.profile.optimal_num_envs_per_worker}")
        print(f"   Batch size: {self.optimizer.profile.optimal_batch_size}")
        print(f"   GPU acceleration: {'ENABLED' if self.optimizer.profile.gpu_available else 'DISABLED'}")

        # Initialize Ray with optimizations
        ray_config = self.optimizer.optimize_ray_init()
        ray.init(**ray_config)

        try:
            # Run training
            from ray import tune

            results = tune.run(
                "PPO",
                name=self.experiment_name,
                config=train_config["algorithm"].to_dict(),
                stop=train_config["stop_criteria"],
                checkpoint_config=train_config["checkpoint_config"],
                verbose=train_config["verbose"],

                # Performance monitoring
                callbacks=[OptimizedCallbacks()],
            )

            return results

        finally:
            ray.shutdown()


from ray.rllib.algorithms.callbacks import DefaultCallbacks

class OptimizedCallbacks(DefaultCallbacks):
    """Callbacks for performance monitoring."""

    def on_train_result(self, *, algorithm, result, **kwargs):
        """Add performance metrics to results."""
        # Calculate samples per second
        if "num_env_steps_sampled" in result:
            samples = result["num_env_steps_sampled"]
            time = result.get("time_total_s", 1)
            result["samples_per_second"] = samples / time

        # GPU utilization
        if torch.cuda.is_available():
            result["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / 1e6
            result["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / 1e6


# Example usage showing performance difference
if __name__ == "__main__":
    import json

    print("🏎️  MARL Hardware Optimization Module")
    print("=" * 60)

    # Create optimizer
    optimizer = HardwareOptimizer()

    # Show Ray init config
    print("\n📡 Optimal Ray Configuration:")
    ray_config = optimizer.optimize_ray_init()
    print(json.dumps(ray_config, indent=2))

    # Show training throughput estimates
    print("\n📊 Expected Training Throughput:")

    total_envs = optimizer.profile.optimal_num_workers * optimizer.profile.optimal_num_envs_per_worker
    steps_per_env = 200  # Typical episode length

    # Conservative estimates
    if optimizer.profile.gpu_available:
        env_steps_per_sec = total_envs * 50  # 50 Hz per env with GPU
        train_samples_per_sec = 10000  # GPU can process this many
    else:
        env_steps_per_sec = total_envs * 30  # 30 Hz per env CPU only
        train_samples_per_sec = 2000  # CPU training throughput

    print(f"  Environment steps/second: ~{env_steps_per_sec:,}")
    print(f"  Training samples/second: ~{train_samples_per_sec:,}")
    print(f"  Estimated time to 1M steps: {1_000_000 / env_steps_per_sec / 60:.1f} minutes")

    # Compare with baseline
    print("\n📈 Performance Improvement vs Baseline:")
    baseline_envs = 4
    baseline_steps_per_sec = baseline_envs * 20

    improvement = env_steps_per_sec / baseline_steps_per_sec
    print(f"  Environment throughput: {improvement:.1f}x faster")
    print(f"  Time to complete experiment: {1/improvement:.1f}x")

    print("\n✅ Your hardware is capable of high-performance MARL research!")
    print("   Use OptimizedTrainingPipeline for maximum throughput.")
