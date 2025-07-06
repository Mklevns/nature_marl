#!/usr/bin/env python3
"""
Example Usage: Unified Nature-Inspired Training Configuration

This example demonstrates how to use the merged training configuration module
for bio-inspired multi-agent reinforcement learning with automatic hardware optimization.
"""

import ray
import gymnasium as gym
from ray import tune
from ray.tune.registry import register_env

# Import the unified training configuration module
from training_config import (
    NatureInspiredTrainingFactory,
    print_nature_inspired_hardware_summary,
    EmergentBehaviorTracker
)

# Example environment (you would replace this with your actual environment)
def simple_nature_env_creator(config):
    """Create a simple multi-agent environment for demonstration."""
    # This is a placeholder - replace with your actual environment
    from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole
    return MultiAgentCartPole(config)


def main():
    """Demonstrate the unified training configuration system."""

    print("🌿 Nature-Inspired Multi-Agent Reinforcement Learning")
    print("=" * 60)

    # Step 1: Analyze the computational ecosystem
    print("\n🔍 Step 1: Analyzing Computational Ecosystem")
    print_nature_inspired_hardware_summary()

    # Step 2: Create the bio-inspired training factory
    print("\n🧬 Step 2: Initializing Bio-Inspired Training Factory")
    factory = NatureInspiredTrainingFactory()

    # Step 3: Register environment
    print("\n🌍 Step 3: Preparing Training Environment")
    register_env("nature_marl_env", simple_nature_env_creator)

    # Get sample environment for space configuration
    sample_env = simple_nature_env_creator({"num_agents": 4})
    obs_space = sample_env.observation_space
    act_space = sample_env.action_space

    # Step 4: Create adaptive training configuration
    print("\n⚙️  Step 4: Creating Adaptive Training Configuration")
    training_config = factory.create_adaptive_config(
        obs_space=obs_space,
        act_space=act_space,
        env_config={
            "num_agents": 4,
            "debug": False,
            "communication_enabled": True
        }
    )

    # Step 5: Get Ray initialization configuration
    print("\n🚀 Step 5: Configuring Distributed Ecosystem")
    ray_config = factory.get_ray_initialization_config()

    # Step 6: Initialize Ray with bio-inspired optimization
    if not ray.is_initialized():
        ray.init(**ray_config)

    # Step 7: Configure experiment parameters
    print("\n🧪 Step 7: Setting Up Evolutionary Experiment")
    tune_config = factory.get_tune_config(
        num_iterations=50,
        experiment_name="nature_inspired_marl_demo"
    )

    # Step 8: Display configuration summary
    print("\n📋 Training Configuration Summary:")
    hardware = factory.hardware_profile
    print(f"   Neural Processing: {hardware.gpu_name}")
    print(f"   Worker Colonies: {hardware.optimal_num_workers}")
    print(f"   Environments per Colony: {hardware.optimal_num_envs_per_worker}")
    print(f"   Neural Batch Size: {hardware.optimal_batch_size}")
    print(f"   Neural Plasticity Rate: {hardware.recommended_learning_rate}")
    print(f"   Communication Channels: {hardware.communication_channels}")
    print(f"   Memory Consolidation Frequency: {hardware.memory_consolidation_frequency}")

    # Step 9: Create complete training pipeline configuration
    print("\n🔧 Step 9: Finalizing Training Pipeline")
    pipeline_config = factory.create_optimized_training_pipeline("nature_marl_demo")

    # Update training config with environment
    training_config = training_config.environment("nature_marl_env")

    print("\n✅ Bio-Inspired Training System Ready!")
    print("   🧠 Neural networks optimized for emergent behavior")
    print("   🐜 Swarm intelligence tracking enabled")
    print("   📡 Pheromone communication analysis active")
    print("   🌱 Adaptive neural plasticity configured")

    # Optional: Run actual training (commented out for example)
    """
    print("\n🚀 Starting Nature-Inspired Training...")

    results = tune.run(
        "PPO",
        config=training_config.to_dict(),
        stop=tune_config["stop"],
        checkpoint_config=tune_config["checkpoint_config"],
        name=tune_config["name"],
        verbose=tune_config["verbose"]
    )

    print("🎉 Training Complete!")
    print(f"Best result: {results.get_best_result()}")
    """

    # Step 10: Demonstrate callback system
    print("\n📊 Step 10: Bio-Inspired Metrics Tracking")
    tracker = EmergentBehaviorTracker()

    print("   Available Emergent Behavior Metrics:")
    print("   - Pheromone communication entropy")
    print("   - Swarm coordination index")
    print("   - Neural plasticity adaptation rate")
    print("   - Memory consolidation strength")
    print("   - Collective intelligence emergence")
    print("   - Emergent strategy complexity")

    # Cleanup
    if ray.is_initialized():
        ray.shutdown()

    print("\n🌿 Nature-Inspired MARL Demo Complete!")
    print("   Your system is ready for bio-inspired multi-agent research.")


def demonstrate_advanced_features():
    """Demonstrate advanced features of the unified training system."""

    print("\n🔬 Advanced Bio-Inspired Features Demonstration")
    print("=" * 50)

    # Create factory with automatic optimization
    factory = NatureInspiredTrainingFactory()

    # Access detailed hardware profile
    profile = factory.hardware_profile

    print(f"\n🧬 Adaptive Neural Parameters:")
    print(f"   Memory Replay Capacity: {profile.memory_replay_capacity:,}")
    print(f"   Communication Channels: {profile.communication_channels}")
    print(f"   Memory Consolidation Frequency: {profile.memory_consolidation_frequency}")
    print(f"   Adaptive Entropy Coefficient: {profile.adaptive_entropy_coeff}")

    # Demonstrate different configuration modes
    print(f"\n⚙️  Configuration Adaptability:")

    # Force GPU configuration (if available)
    if profile.gpu_available:
        print("   🧠 GPU Neural Network Configuration Available")
        print(f"      - High-performance neural processing: {profile.gpu_memory_gb:.1f}GB")
        print(f"      - Advanced synaptic computation: {profile.gpu_compute_capability}")

    # CPU coordination configuration
    print("   🔗 CPU Coordination Network Configuration")
    print(f"      - Distributed coordination cores: {profile.cpu_cores_physical}")
    print(f"      - Parallel processing threads: {profile.cpu_cores_logical}")

    # Memory-based adaptations
    print(f"\n🧮 Memory-Based Neural Adaptations:")
    print(f"   - Available neural memory: {profile.ram_available_gb:.1f}GB")
    print(f"   - Memory-efficient batch processing: {profile.optimal_batch_size}")
    print(f"   - Adaptive replay buffer: {profile.memory_replay_capacity:,} experiences")

    # Ray ecosystem configuration
    ray_config = factory.get_ray_initialization_config()
    print(f"\n🌐 Distributed Ecosystem Configuration:")
    print(f"   - Coordination CPUs: {ray_config['num_cpus']}")
    print(f"   - Neural Processing GPUs: {ray_config.get('num_gpus', 0)}")
    print(f"   - Shared Memory Pool: {ray_config['object_store_memory'] / 1e9:.1f}GB")


if __name__ == "__main__":
    # Run the main demonstration
    main()

    # Show advanced features
    demonstrate_advanced_features()

    print("\n" + "="*60)
    print("🎓 Integration Guide:")
    print("   1. Replace your existing training config imports")
    print("   2. Use NatureInspiredTrainingFactory for all configurations")
    print("   3. Enable EmergentBehaviorTracker for bio-inspired metrics")
    print("   4. Leverage adaptive configuration for optimal performance")
    print("   5. Monitor pheromone communication and swarm coordination")
    print("="*60)
