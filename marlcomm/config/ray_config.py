# ray_config_final.py - Final Fixed Ray 2.9.0 Configuration
"""
Final fixed version that properly handles Ray 2.9.0 initialization issues.
This version is conservative and only uses parameters that work reliably.
"""

import os
import psutil
import torch
import time
from typing import Dict, Any, Optional
from ray.rllib.algorithms.ppo import PPOConfig


class NatureInspiredRayConfig:
    """Fixed Ray configuration for Nature-Inspired MARL project"""

    # Ray 2.9.0 Feature Flags
    USE_NEW_RL_MODULE_API = False  # Ray 2.9 supports this!
    USE_NEW_LEARNER_API = False    # Ray 2.9 supports this!
    USE_CONNECTORS = True         # Recommended for Ray 2.9

    @staticmethod
    def check_dashboard_available() -> bool:
        """Properly check if Ray dashboard is available"""
        try:
            # First check if we can import it
            import ray.dashboard

            # Then check if ray[default] is actually installed
            import subprocess
            result = subprocess.run(
                ["pip", "show", "ray"],
                capture_output=True,
                text=True
            )

            # Check if ray[default] extras are installed
            if "ray[default]" in result.stdout or "aiohttp" in result.stdout:
                return True

            # Try to import key dashboard dependencies
            try:
                import aiohttp
                import aioredis
                return True
            except ImportError:
                return False

        except ImportError:
            return False

    @staticmethod
    def get_ray_init_config(
        use_nvme: bool = False,
        nvme_base_path: str = "/mnt/d/MLData",
        num_gpus: int = 1,
        dashboard: bool = False
    ) -> Dict[str, Any]:
        """
        Get Ray initialization config that actually works with Ray 2.9.0

        Args:
            use_nvme: Whether to use NVMe storage for temp files
            nvme_base_path: Base path for NVMe storage
            num_gpus: Number of GPUs to use
            dashboard: Whether to enable dashboard

        Returns:
            Dictionary of ray.init() parameters
        """
        # Start with minimal configuration that works
        config = {
            "num_cpus": psutil.cpu_count(),
            "num_gpus": num_gpus,
            "logging_level": "WARNING",
        }

        # Add memory configuration (this works)
        total_memory = psutil.virtual_memory().total
        config["object_store_memory"] = int(total_memory * 0.3)  # 30% for object store

        # NVMe optimization - be careful with temp directory
        if use_nvme:
            nvme_temp = os.path.join(nvme_base_path, "ray_temp")
            nvme_spill = os.path.join(nvme_base_path, "spill")

            # Create directories
            for dir_path in [nvme_temp, nvme_spill]:
                os.makedirs(dir_path, exist_ok=True)

            # Set environment variables BEFORE ray.init
            os.environ["RAY_TMPDIR"] = nvme_temp
            os.environ["TMPDIR"] = nvme_temp

            # Configure object spilling via environment variable
            os.environ["RAY_object_spilling_config"] = (
                '{"type":"filesystem","params":{"directory_path":"' + nvme_spill + '"}}'
            )

            # Only add _temp_dir if we're sure it won't cause timeouts
            # For now, let's rely on environment variables instead
            # config["_temp_dir"] = nvme_temp  # This can cause timeouts

        # Dashboard configuration - only if truly available
        if dashboard and NatureInspiredRayConfig.check_dashboard_available():
            config["include_dashboard"] = True
            config["dashboard_host"] = "0.0.0.0"
        elif dashboard:
            print("⚠️  Dashboard requested but ray[default] not properly installed")
            print("   Install with: pip install 'ray[default]'")
            print("   Continuing without dashboard...")

        # Add node IP for stability
        config["_node_ip_address"] = "127.0.0.1"

        return config

    @staticmethod
    def safe_ray_init(**kwargs) -> bool:
        """Safely initialize Ray with proper cleanup"""
        import ray

        # First, ensure any existing Ray instance is stopped
        try:
            ray.shutdown()
            time.sleep(1)
        except:
            pass

        # Force stop any zombie processes
        os.system("ray stop --force 2>/dev/null")
        time.sleep(2)

        # Now try to initialize
        try:
            ray.init(**kwargs)
            return True
        except Exception as e:
            print(f"❌ Ray init failed: {e}")
            return False

    @staticmethod
    def get_ppo_config(
        env_name: str,
        env_config: Dict[str, Any],
        num_workers: int = 4,
        num_envs_per_worker: int = 16,
        train_batch_size: int = 32768,
        use_gpu: bool = True,
        model_config: Optional[Dict[str, Any]] = None
    ) -> PPOConfig:
        """
        Get PPO configuration optimized for Ray 2.9.0
        """
        config = PPOConfig()

        # Environment configuration
        config.environment(
            env=env_name,
            disable_env_checking=True,
            env_config=env_config
        )

        # Framework
        config.framework("torch")

        # Resources
        config.resources(
            num_gpus=1 if use_gpu else 0,
            num_cpus_per_worker=1,
            num_gpus_per_worker=0,  # Workers use CPU
            placement_strategy="SPREAD"
        )

        # Rollouts configuration
        config.rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=num_envs_per_worker,
            rollout_fragment_length=512,
            batch_mode="truncate_episodes",
            remote_worker_envs=True,
            # Enable connectors for Ray 2.9.0
            enable_connectors=NatureInspiredRayConfig.USE_CONNECTORS,
            # Compression for efficiency
            compress_observations=True
        )

        # Training configuration
        train_config = {
            "train_batch_size": train_batch_size,
            "sgd_minibatch_size": min(4096, train_batch_size // 4),
            "num_sgd_iter": 10,
            "lr": 0.0003,
            "gamma": 0.99,
            "lambda_": 0.95,
            "clip_param": 0.2,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "grad_clip": 10.0
        }

        # Model configuration
        if model_config:
            train_config["model"] = model_config
        else:
            # Default bio-inspired model config
            train_config["model"] = {
                "fcnet_hiddens": [512, 512, 256],
                "fcnet_activation": "relu",
                "use_lstm": False,
                "max_seq_len": 20,
                "lstm_cell_size": 256,
                "lstm_use_prev_action": True,
                "lstm_use_prev_reward": True,
                # Custom model support
                "custom_model": model_config.get("custom_model") if model_config else None,
                "custom_model_config": model_config.get("custom_model_config", {}) if model_config else {},
            }

        config.training(**train_config)

        # Reporting configuration
        config.reporting(
            keep_per_episode_custom_metrics=True,
            min_time_s_per_iteration=0,
            min_sample_timesteps_per_iteration=1000
        )

        # Debugging
        config.debugging(
            log_level="ERROR",
            seed=42
        )

        # Fault tolerance for Ray 2.9.0
        config.evaluation(
            evaluation_interval=1,            
            evaluation_num_workers=1
        )

        config.fault_tolerance(
            recreate_failed_workers=True,
            max_num_worker_restarts=3,
        )

        # Checkpointing
        config.checkpointing(
            export_native_model_files=True,
            checkpoint_trainable_policies_only=False
        )

        # Ray 2.9.0 experimental features
        config.experimental(
            _disable_preprocessor_api=False,
            _disable_action_flattening=False,
            _disable_execution_plan_api=True  # Use new training stack
        )

        return config

    @staticmethod
    def validate_ray_installation() -> bool:
        """Validate Ray installation and components"""
        print("🔍 Validating Ray 2.9.0 Installation...")
        print("=" * 50)

        issues = []

        # Check Ray version
        try:
            import ray
            version = ray.__version__
            major, minor = map(int, version.split('.')[:2])
            print(f"✅ Ray version: {version}")

            if major < 2 or (major == 2 and minor < 9):
                issues.append(f"Ray {version} is older than 2.9.0")
        except ImportError:
            issues.append("Ray is not installed")
            return False

        # Check RLlib
        try:
            import ray.rllib
            print("✅ Ray RLlib is available")
        except ImportError:
            issues.append("Ray RLlib is not available")

        # Check Tune
        try:
            import ray.tune
            print("✅ Ray Tune is available")
        except ImportError:
            issues.append("Ray Tune is not available")

        # Check for dashboard properly
        if NatureInspiredRayConfig.check_dashboard_available():
            print("✅ Ray Dashboard is properly installed")
        else:
            print("⚠️  Ray Dashboard not fully installed")
            print("   To install: pip install 'ray[default]'")

        # Check GPU
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            print(f"✅ GPU detected: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")
        else:
            print("⚠️  No GPU detected")

        if issues:
            print("\n❌ Issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False

        print("\n✅ Ray 2.9.0 installation is valid!")
        return True


# Convenience functions for quick setup
def get_ray_init_kwargs(**kwargs) -> Dict[str, Any]:
    """Get Ray init kwargs with defaults"""
    return NatureInspiredRayConfig.get_ray_init_config(**kwargs)


def get_ppo_config_for_nature_marl(
    env_name: str = "nature_marl_comm",
    n_agents: int = 8,
    grid_size: tuple = (12, 12),
    **kwargs
) -> PPOConfig:
    """Get PPO config for Nature-Inspired MARL"""
    env_config = {
        "env_type": kwargs.pop("env_type", "foraging"),
        "n_agents": n_agents,
        "grid_size": grid_size,
        "episode_length": kwargs.pop("episode_length", 100),
        "render_mode": kwargs.pop("render_mode", None)
    }

    return NatureInspiredRayConfig.get_ppo_config(
        env_name=env_name,
        env_config=env_config,
        **kwargs
    )


def safe_ray_init(**kwargs) -> bool:
    """Safely initialize Ray"""
    return NatureInspiredRayConfig.safe_ray_init(**kwargs)


if __name__ == "__main__":
    # Clean any existing Ray processes first
    print("🧹 Cleaning up any existing Ray processes...")
    os.system("ray stop --force 2>/dev/null")
    time.sleep(2)

    # Validate installation
    NatureInspiredRayConfig.validate_ray_installation()

    # Test configuration
    print("\n🧪 Testing Ray initialization...")
    import ray

    try:
        ray_config = get_ray_init_kwargs(
            use_nvme=True,
            dashboard=False  # Don't try dashboard if not installed
        )

        if safe_ray_init(**ray_config):
            print("✅ Ray initialization successful!")

            resources = ray.cluster_resources()
            print("\n📊 Ray Cluster Resources:")
            for resource, amount in resources.items():
                print(f"   {resource}: {amount}")

            ray.shutdown()
        else:
            print("❌ Ray initialization failed")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
