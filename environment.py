# File: marlcomm/enviroment.py
"""
Multi-Agent Environment Setup for Nature-Inspired Communication

This module handles the creation and configuration of PettingZoo multi-agent
environments, specifically the simple_spread scenario where agents must
coordinate to cover landmarks efficiently.
"""

import warnings
from typing import Dict, Any
from pettingzoo.mpe.simple_spread_v3 import parallel_env
import supersuit as ss
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# Suppress pygame warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")


class EnvironmentConfig:
    """Configuration class for multi-agent environment settings."""

    def __init__(self,
                 num_agents: int = 3,
                 max_cycles: int = 25,
                 local_ratio: float = 0.5,
                 continuous_actions: bool = False,
                 render_mode: str = None):
        """
        Initialize environment configuration.

        Args:
            num_agents: Number of agents in the environment
            max_cycles: Maximum number of environment steps per episode
            local_ratio: Ratio of local vs global rewards
            continuous_actions: Whether to use continuous action space
            render_mode: Rendering mode (None, "human", "rgb_array")
        """
        self.num_agents = num_agents
        self.max_cycles = max_cycles
        self.local_ratio = local_ratio
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "N": self.num_agents,
            "max_cycles": self.max_cycles,
            "local_ratio": self.local_ratio,
            "continuous_actions": self.continuous_actions,
            "render_mode": self.render_mode
        }


def create_base_environment(config: EnvironmentConfig) -> ParallelPettingZooEnv:
    """
    Create the base PettingZoo environment.

    Args:
        config: Environment configuration object

    Returns:
        Configured PettingZoo parallel environment
    """
    # Create the Multi-Agent Particle Environment
    env = parallel_env(**config.to_dict())
    return env


def apply_environment_wrappers(env) -> ParallelPettingZooEnv:
    """
    Apply SuperSuit wrappers for preprocessing and compatibility.

    Args:
        env: Base PettingZoo environment

    Returns:
        Wrapped environment ready for RLlib
    """
    # Remove agents that die during episode
    env = ss.black_death_v3(env)

    # Ensure consistent observation shapes across agents
    env = ss.pad_observations_v0(env)

    # Note: Avoiding pad_action_space_v0 due to compatibility issues
    # with the new RLlib API stack

    return env


def convert_to_rllib_env(env) -> ParallelPettingZooEnv:
    """
    Convert PettingZoo environment to RLlib-compatible format.

    Args:
        env: Wrapped PettingZoo environment

    Returns:
        RLlib-compatible environment
    """
    return ParallelPettingZooEnv(env)


def create_nature_comm_env(env_config: Dict[str, Any]) -> ParallelPettingZooEnv:
    """
    Complete environment creation pipeline for nature-inspired communication.

    This is the main function used by RLlib for environment creation.

    Args:
        env_config: Dictionary containing environment configuration

    Returns:
        Fully configured RLlib-compatible multi-agent environment
    """
    debug = env_config.get("debug", False)

    # Extract configuration parameters
    config = EnvironmentConfig(
        num_agents=env_config.get("num_agents", 3),
        max_cycles=env_config.get("max_cycles", 25),
        local_ratio=env_config.get("local_ratio", 0.5),
        continuous_actions=env_config.get("continuous_actions", False),
        render_mode=env_config.get("render_mode", None)
    )

    if debug:
        print(f"🌱 Creating environment with config: {config.to_dict()}")

    try:
        # Step 1: Create base environment
        base_env = create_base_environment(config)

        # Step 2: Apply preprocessing wrappers
        wrapped_env = apply_environment_wrappers(base_env)

        # Step 3: Convert to RLlib format
        rllib_env = convert_to_rllib_env(wrapped_env)

        if debug:
            print(f"🌿 Environment created successfully!")
            print(f"   Agents: {len(base_env.possible_agents)}")
            print(f"   Observation space: {rllib_env.observation_space}")
            print(f"   Action space: {rllib_env.action_space}")

        return rllib_env

    except Exception as e:
        if debug:
            print(f"❌ Environment creation failed: {e}")
        raise


def register_nature_comm_environment():
    """
    Register the environment with Ray Tune for easy reference.

    This allows using "nature_marl_comm" as a string identifier
    in RLlib configurations.
    """
    register_env("nature_marl_comm", create_nature_comm_env)
    print("📝 Registered 'nature_marl_comm' environment")


def test_environment_creation(debug: bool = True) -> bool:
    """
    Test environment creation to verify everything works correctly.

    Args:
        debug: Whether to print debug information

    Returns:
        True if environment creation succeeds, False otherwise
    """
    try:
        print("🧪 Testing environment creation...")

        # Test with default configuration
        test_config = {"debug": debug}
        env = create_nature_comm_env(test_config)

        # Test basic functionality
        obs, info = env.reset()
        print(f"✅ Environment reset successful. Agents: {list(obs.keys())}")

        # Test a few steps
        for i in range(3):
            actions = {}
            for agent in obs.keys():
                if hasattr(env.action_space, 'spaces') and agent in env.action_space.spaces:
                    actions[agent] = env.action_space.spaces[agent].sample()
                else:
                    # Fallback action sampling
                    sample = env.action_space.sample()
                    actions[agent] = sample[agent] if isinstance(sample, dict) else sample

            obs, rewards, dones, truncs, info = env.step(actions)
            print(f"   Step {i+1}: Rewards = {rewards}")

            if all(dones.values()) or all(truncs.values()):
                obs, info = env.reset()
                break

        env.close()
        print("✅ Environment test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


if __name__ == "__main__":
    # Run environment test when executed directly
    test_environment_creation(debug=True)
