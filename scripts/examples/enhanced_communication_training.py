# File: nature_marl/training/examples/enhanced_communication_training.py
#!/usr/bin/env python3

"""
Example usage of the enhanced nature-inspired communication module
with custom loss integration and proper multi-agent configuration.
"""

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune.registry import register_env
from pettingzoo.sisl import waterworld_v4
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
import supersuit as ss

# Import your enhanced module
from marlcomm.rl_module import (
    NatureInspiredCommModule,
    get_nature_comm_config,
    create_enhanced_nature_comm_module_spec
)


def create_communication_environment(config):
    """Create a preprocessed multi-agent environment suitable for communication."""
    env = waterworld_v4.parallel_env(
        n_pursuers=config.get("num_agents", 8),
        n_evaders=5,
        n_poison=10,
        max_cycles=500
    )

    # Apply preprocessing
    env = ss.color_reduction_v0(env, mode='B')  # Grayscale
    env = ss.resize_v1(env, x_size=64, y_size=64)  # Resize
    env = ss.frame_stack_v1(env, 3)  # Frame stacking for temporal info
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)  # Normalize

    return env


class CustomPPOLearner:
    """
    Custom PPO learner that integrates communication loss.
    This would typically be implemented by subclassing PPOTorchLearner.
    """

    @staticmethod
    def add_communication_loss(policy_loss, model, config):
        """Add communication regularization to the standard PPO loss."""
        if hasattr(model, 'get_communication_loss'):
            comm_loss_coeff = config.get("comm_loss_coeff", 0.1)
            comm_loss = model.get_communication_loss(
                entropy_coeff=config.get("comm_entropy_coeff", 0.01),
                sparsity_coeff=config.get("comm_sparsity_coeff", 0.001),
                diversity_coeff=config.get("comm_diversity_coeff", 0.01)
            )
            total_loss = policy_loss + comm_loss_coeff * comm_loss
            return total_loss
        return policy_loss


def main():
    """Main training function with enhanced communication."""

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Environment configuration
    num_agents = 8
    env_config = {"num_agents": num_agents}

    # Register environment
    register_env("waterworld_comm",
                lambda config: PettingZooEnv(create_communication_environment(config)))

    # Get enhanced communication configuration
    model_config = get_nature_comm_config(
        num_agents=num_agents,
        embed_dim=256,
        comm_rounds=3,
        attention_heads=8
    )

    # Create sample environment to get spaces
    sample_env = PettingZooEnv(create_communication_environment(env_config))
    obs_space = list(sample_env.observation_spaces.values())[0]
    action_space = list(sample_env.action_spaces.values())[0]

    # Enhanced PPO configuration
    config = (
        PPOConfig()
        .environment(
            env="waterworld_comm",
            env_config=env_config
        )
        .framework("torch")
        .training(
            # Standard PPO hyperparameters
            train_batch_size=8192,
            sgd_minibatch_size=512,
            num_sgd_iter=10,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,

            # Custom communication loss coefficients
            comm_loss_coeff=0.1,
            comm_entropy_coeff=0.01,
            comm_sparsity_coeff=0.001,
            comm_diversity_coeff=0.01,

            # Model configuration
            model=model_config
        )
        .multi_agent(
            # Use parameter sharing for homogeneous agents
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
            policies_to_train=["shared_policy"]
        )
        .rl_module(
            rl_module_spec=create_enhanced_nature_comm_module_spec(
                obs_space=obs_space,
                act_space=action_space,
                model_config=model_config
            )
        )
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=2
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=1 if ray.get_gpu_ids() else 0
        )
        .resources(
            num_gpus=1 if ray.get_gpu_ids() else 0
        )
        .debugging(
            log_level="INFO"
        )
    )

    # Create stop criteria
    stop_criteria = {
        "training_iteration": 200,
        "episode_reward_mean": 50.0,  # Environment-specific target
        "time_total_s": 7200  # 2 hours max
    }

    # Run training
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=ray.air.RunConfig(
            name="enhanced_nature_comm_experiment",
            stop=stop_criteria,
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=10,
                num_to_keep=3
            ),
            verbose=1
        )
    )

    results = tuner.fit()

    # Get best result
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Best mean reward: {best_result.metrics['episode_reward_mean']}")
    print(f"Best checkpoint: {best_result.checkpoint}")

    ray.shutdown()


if __name__ == "__main__":
    main()
