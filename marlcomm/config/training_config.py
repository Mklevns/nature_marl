# marlcomm/config/training_config.py
"""
Modern training configuration using RLlib 2.9.x fluent API.
"""

from typing import Dict, Any, Optional
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.tune.registry import register_env

# Import your models from rl_module
from marlcomm.models import BioInspiredRLModule
from marlcomm.environments.emergence_environments import (
    CommunicationEnv,  # Adjust based on your actual environment names
)


def create_bio_inspired_ppo_config(
    env_name: str = "CommEnv-v0",
    num_workers: int = 4,
    num_gpus: int = 1,
    framework: str = "torch",
    experiment_config: Optional[Dict[str, Any]] = None
) -> PPOConfig:
    """
    Create a modern PPOConfig for bio-inspired multi-agent training.

    Args:
        env_name: Name of the registered environment
        num_workers: Number of parallel rollout workers
        num_gpus: Number of GPUs to use
        framework: Deep learning framework ("torch" or "tf2")
        experiment_config: Additional experiment-specific settings

    Returns:
        Configured PPOConfig object ready for training
    """

    exp_config = experiment_config or {}

    # Bio-inspired parameters
    use_pheromones = exp_config.get("use_pheromones", True)
    use_plasticity = exp_config.get("use_plasticity", True)
    comm_channel_dim = exp_config.get("comm_channel_dim", 16)
    plasticity_rate = exp_config.get("plasticity_rate", 0.001)
    hidden_dim = exp_config.get("hidden_dim", 64)

    # Create base PPO configuration with fluent API
    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config={
                "grid_size": exp_config.get("grid_size", (10, 10)),
                "n_agents": exp_config.get("n_agents", 2),
                "episode_limit": exp_config.get("episode_limit", 100),
                "reward_type": exp_config.get("reward_type", "sparse"),
                "enable_pheromones": use_pheromones,
            },
            render_env=False,
            clip_rewards=exp_config.get("clip_rewards", None),
            normalize_actions=False,
        )
        .rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=exp_config.get("num_envs_per_worker", 1),
            batch_mode="complete_episodes",
            rollout_fragment_length="auto",
            enable_connectors=True,
            compress_observations=False,
            sampler_perf_stats_ema_coef=None,
        )
        .training(
            # PPO-specific hyperparameters
            lambda_=0.95,
            gamma=exp_config.get("gamma", 0.99),
            lr=exp_config.get("learning_rate", 3e-4),
            num_sgd_iter=exp_config.get("num_sgd_iter", 10),
            sgd_minibatch_size=exp_config.get("sgd_minibatch_size", 128),
            train_batch_size=exp_config.get("train_batch_size", 4000),
            clip_param=exp_config.get("clip_param", 0.2),
            vf_clip_param=exp_config.get("vf_clip_param", 10.0),
            entropy_coeff=exp_config.get("entropy_coeff", 0.01),
            vf_loss_coeff=exp_config.get("vf_loss_coeff", 0.5),
            kl_target=exp_config.get("kl_target", 0.01),
            kl_coeff=exp_config.get("kl_coeff", 0.2),

            # Gradient settings
            grad_clip=exp_config.get("grad_clip", 0.5),
            grad_clip_by="global_norm",

            # Learning rate schedule (optional)
            lr_schedule=exp_config.get("lr_schedule", None) or [
                [0, exp_config.get("learning_rate", 3e-4)],
                [exp_config.get("total_timesteps", 1e6), 1e-5]
            ],
        )
        .resources(
            num_gpus=num_gpus,
            num_cpus_per_worker=1,
            num_gpus_per_learner_worker=num_gpus / max(1, num_workers) if num_gpus > 0 else 0,
            custom_resources_per_worker={},
        )
        .framework(
            framework=framework,
            eager_tracing=False,
        )
        .debugging(
            seed=exp_config.get("seed", 42),
            log_level=exp_config.get("log_level", "INFO"),
            log_sys_usage=True,
        )
        .reporting(
            keep_per_episode_custom_metrics=True,
            metrics_num_episodes_for_smoothing=exp_config.get("smoothing_episodes", 10),
            min_sample_timesteps_per_iteration=1000,
            min_time_s_per_iteration=None,
        )
        .experimental(
            _enable_new_api_stack=True,
            _disable_preprocessor_api=True,
            _disable_action_flattening=True,
        )
        .fault_tolerance(
            recreate_failed_workers=True,
            max_num_worker_restarts=1000,
            delay_between_worker_restarts_s=60.0,
        )
    )

    # Configure multi-agent setup if needed
    if exp_config.get("multi_agent", True):
        # Model configuration for each agent type
        speaker_model_config = {
            "is_speaker": True,
            "comm_channel_dim": comm_channel_dim,
            "use_pheromones": use_pheromones,
            "use_plasticity": use_plasticity,
            "plasticity_rate": plasticity_rate,
            "hidden_dim": hidden_dim,
            "grid_size": exp_config.get("grid_size", (10, 10)),
        }

        listener_model_config = {
            "is_speaker": False,
            "comm_channel_dim": comm_channel_dim,
            "use_pheromones": use_pheromones,
            "use_plasticity": use_plasticity,
            "plasticity_rate": plasticity_rate,
            "hidden_dim": hidden_dim,
            "grid_size": exp_config.get("grid_size", (10, 10)),
        }

        # Set up multi-agent configuration
        config = config.multi_agent(
            # Policy mapping function
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: (
                "speaker" if agent_id.startswith("speaker") else "listener"
            ),

            # Define which policies to train
            policies_to_train=exp_config.get("policies_to_train", ["speaker", "listener"]),

            # Count steps by agent
            count_steps_by="agent_steps",
        )

        # Set up RLModule specs for multi-agent
        config = config.rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    "speaker": SingleAgentRLModuleSpec(
                        module_class=BioInspiredRLModule,
                        model_config_dict=speaker_model_config,
                    ),
                    "listener": SingleAgentRLModuleSpec(
                        module_class=BioInspiredRLModule,
                        model_config_dict=listener_model_config,
                    ),
                }
            ),
        )
    else:
        # Single agent configuration
        config = config.rl_module(
            rl_module_spec=SingleAgentRLModuleSpec(
                module_class=BioInspiredRLModule,
                model_config_dict={
                    "is_speaker": exp_config.get("is_speaker", True),
                    "comm_channel_dim": comm_channel_dim,
                    "use_pheromones": use_pheromones,
                    "use_plasticity": use_plasticity,
                    "plasticity_rate": plasticity_rate,
                    "hidden_dim": hidden_dim,
                    "grid_size": exp_config.get("grid_size", (10, 10)),
                }
            ),
        )

    # Add custom callbacks if provided
    if exp_config.get("callbacks_class"):
        config = config.callbacks(exp_config["callbacks_class"])

    return config


# Predefined experiment configurations
EXPERIMENT_CONFIGS = {
    "baseline": {
        "name": "Baseline (No Bio-Inspired)",
        "use_pheromones": False,
        "use_plasticity": False,
        "comm_channel_dim": 16,
        "n_agents": 2,
        "grid_size": (10, 10),
        "episode_limit": 100,
        "total_timesteps": 1e6,
        "learning_rate": 3e-4,
        "train_batch_size": 4000,
    },

    "pheromones_only": {
        "name": "Pheromone Communication Only",
        "use_pheromones": True,
        "use_plasticity": False,
        "comm_channel_dim": 16,
        "n_agents": 4,
        "grid_size": (20, 20),
        "episode_limit": 200,
        "total_timesteps": 2e6,
        "learning_rate": 3e-4,
        "entropy_coeff": 0.01,
    },

    "plasticity_only": {
        "name": "Neural Plasticity Only",
        "use_pheromones": False,
        "use_plasticity": True,
        "plasticity_rate": 0.001,
        "comm_channel_dim": 16,
        "n_agents": 2,
        "grid_size": (10, 10),
        "episode_limit": 100,
        "total_timesteps": 1e6,
        "learning_rate": 3e-4,
    },

    "full_bio_inspired": {
        "name": "Full Bio-Inspired System",
        "use_pheromones": True,
        "use_plasticity": True,
        "plasticity_rate": 0.001,
        "comm_channel_dim": 32,
        "n_agents": 6,
        "grid_size": (30, 30),
        "episode_limit": 300,
        "total_timesteps": 5e6,
        "learning_rate": 1e-4,
        "entropy_coeff": 0.02,
        "num_sgd_iter": 20,
        "train_batch_size": 8000,
    },

    "emergence_study": {
        "name": "Emergent Communication Study",
        "use_pheromones": True,
        "use_plasticity": True,
        "plasticity_rate": 0.005,
        "comm_channel_dim": 64,
        "n_agents": 8,
        "grid_size": (40, 40),
        "episode_limit": 500,
        "total_timesteps": 1e7,
        "learning_rate": 5e-5,
        "entropy_coeff": 0.05,
        "num_sgd_iter": 30,
        "train_batch_size": 16000,
        "sgd_minibatch_size": 256,
        "num_workers": 8,
    },

    "scalability_test": {
        "name": "Scalability Test",
        "use_pheromones": True,
        "use_plasticity": False,  # Disable for performance
        "comm_channel_dim": 16,
        "n_agents": 16,
        "grid_size": (50, 50),
        "episode_limit": 200,
        "total_timesteps": 5e6,
        "learning_rate": 1e-4,
        "train_batch_size": 32000,
        "sgd_minibatch_size": 512,
        "num_workers": 16,
        "num_envs_per_worker": 2,
    },
}


def get_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """
    Get a predefined experiment configuration.

    Args:
        experiment_name: Name of the experiment configuration

    Returns:
        Dictionary of experiment parameters

    Raises:
        ValueError: If experiment name is not found
    """
    if experiment_name not in EXPERIMENT_CONFIGS:
        available = ", ".join(EXPERIMENT_CONFIGS.keys())
        raise ValueError(
            f"Unknown experiment: '{experiment_name}'. "
            f"Available experiments: {available}"
        )

    return EXPERIMENT_CONFIGS[experiment_name].copy()


def list_experiments():
    """List all available experiment configurations."""
    print("\nAvailable Experiment Configurations:")
    print("-" * 50)
    for name, config in EXPERIMENT_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config.get('name', 'N/A')}")
        print(f"  Agents: {config.get('n_agents', 'N/A')}")
        print(f"  Grid Size: {config.get('grid_size', 'N/A')}")
        print(f"  Bio-Inspired Features:")
        print(f"    - Pheromones: {config.get('use_pheromones', False)}")
        print(f"    - Plasticity: {config.get('use_plasticity', False)}")
        print(f"  Total Timesteps: {config.get('total_timesteps', 'N/A'):,.0f}")


# Example usage function
def create_config_for_experiment(experiment_name: str, **overrides) -> PPOConfig:
    """
    Create a PPOConfig for a specific experiment with optional overrides.

    Example:
        config = create_config_for_experiment(
            "full_bio_inspired",
            num_workers=8,
            num_gpus=2,
            learning_rate=1e-3
        )
    """
    # Get base experiment config
    exp_config = get_experiment_config(experiment_name)

    # Apply overrides
    exp_config.update(overrides)

    # Extract system-level configs
    num_workers = exp_config.pop("num_workers", 4)
    num_gpus = exp_config.pop("num_gpus", 1)
    framework = exp_config.pop("framework", "torch")

    # Create and return PPOConfig
    return create_bio_inspired_ppo_config(
        num_workers=num_workers,
        num_gpus=num_gpus,
        framework=framework,
        experiment_config=exp_config
    )
