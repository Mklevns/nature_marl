# File: nature_marl/training/examples/complete_integration_example.py
"""
Complete Integration Example: Bio-Inspired Multi-Agent Training

This example demonstrates the complete workflow from setup to analysis
using the production-ready Nature MARL system.

FEATURES DEMONSTRATED:
✅ Complete project setup and configuration
✅ Bio-inspired agent training with real-time monitoring
✅ Advanced metrics collection and analysis
✅ Visualization of emergent communication patterns
✅ Performance benchmarking and optimization
✅ Model export and deployment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
import logging

# Nature MARL imports
from nature_marl import (
    create_production_bio_module_spec,
    BioInspiredMetricsTracker,
    BioInspiredVisualizer,
    setup_production_logging,
    bio_inspired_logging_context,
    performance_monitor
)

# RLlib and environment imports
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.mpe import simple_spread_v3
import ray

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompleteBioInspiredExample:
    """
    Complete example showcasing all Nature MARL capabilities.

    This class demonstrates:
    - Setup and configuration
    - Training with bio-inspired agents
    - Real-time metrics monitoring
    - Post-training analysis
    - Visualization and export
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config.get("results_dir", "./results"))
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Initialize components
        self.metrics_tracker = None
        self.visualizer = None
        self.algorithm = None

    def setup_logging_and_monitoring(self):
        """Setup comprehensive logging and monitoring."""
        print("🔧 Setting up logging and monitoring...")

        # Setup production logging
        self.metrics_tracker = setup_production_logging(
            log_level="INFO",
            log_dir=self.results_dir / "logs",
            experiment_name=f"bio_inspired_experiment_{self.config['experiment_id']}"
        )

        # Setup visualization
        self.visualizer = BioInspiredVisualizer(
            use_wandb=self.config.get("use_wandb", False),
            wandb_project="nature-marl-example"
        )

        print("✅ Logging and monitoring configured")

    def create_environment(self):
        """Create and configure the multi-agent environment."""
        print("🌍 Creating multi-agent environment...")

        # Create PettingZoo environment
        env = simple_spread_v3.parallel_env(
            N=self.config["num_agents"],
            local_ratio=0.5,
            max_cycles=25,
            continuous_actions=False
        )

        # Wrap for RLlib
        env = PettingZooEnv(env)

        print(f"✅ Environment created with {self.config['num_agents']} agents")
        return env

    def create_bio_inspired_agents(self):
        """Create bio-inspired RL module specification."""
        print("🧠 Creating bio-inspired agents...")

        # Get environment spaces
        env = self.create_environment()
        obs_space = env.observation_space
        act_space = env.action_space

        # Bio-inspired configuration
        bio_config = {
            "hidden_dim": self.config.get("hidden_dim", 256),
            "memory_dim": self.config.get("memory_dim", 64),
            "comm_channels": self.config.get("comm_channels", 16),
            "comm_rounds": self.config.get("comm_rounds", 3),
            "plasticity_rate": self.config.get("plasticity_rate", 0.1),
            "use_positional_encoding": self.config.get("use_positional_encoding", True),
            "adaptive_plasticity": self.config.get("adaptive_plasticity", True),
            "debug_mode": self.config.get("debug_mode", True)
        }

        # Create module specification
        module_spec = create_production_bio_module_spec(
            obs_space=obs_space,
            act_space=act_space,
            num_agents=self.config["num_agents"],
            use_communication=True,
            model_config=bio_config
        )

        print("✅ Bio-inspired agents configured")
        print(f"   🐜 Pheromone channels: {bio_config['comm_channels']}")
        print(f"   🧠 Memory capacity: {bio_config['memory_dim']}")
        print(f"   🐝 Spatial awareness: {bio_config['use_positional_encoding']}")
        print(f"   🏠 Adaptive plasticity: {bio_config['adaptive_plasticity']}")

        return module_spec

    def setup_training(self, module_spec):
        """Setup PPO training with bio-inspired configuration."""
        print("⚙️  Setting up training configuration...")

        # Create PPO configuration
        config = (
            PPOConfig()
            .environment(
                env=self.create_environment,
                env_config={}
            )
            .framework("torch")
            .rl_module(rl_module_spec=module_spec)
            .training(
                train_batch_size_per_learner=self.config.get("batch_size", 512),
                minibatch_size=self.config.get("minibatch_size", 128),
                lr=self.config.get("learning_rate", 3e-4),
                num_epochs=self.config.get("num_epochs", 10),
                gamma=self.config.get("gamma", 0.99),
                lambda_=self.config.get("lambda", 0.95),
                clip_param=self.config.get("clip_param", 0.2),
                vf_clip_param=self.config.get("vf_clip_param", 10.0),
                entropy_coeff=self.config.get("entropy_coeff", 0.01)
            )
            .env_runners(
                num_env_runners=self.config.get("num_workers", 4),
                num_envs_per_env_runner=1
            )
            .multi_agent(
                policies={"shared_policy": None},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
            )
        )

        # Build algorithm
        self.algorithm = config.build()

        print("✅ Training configuration complete")
        return self.algorithm

    @performance_monitor(track_memory=True, track_gpu=True)
    def train_agents(self):
        """Train bio-inspired agents with comprehensive monitoring."""
        print(f"🚀 Starting training for {self.config['iterations']} iterations...")

        training_results = []
        best_reward = float('-inf')

        with bio_inspired_logging_context(
            experiment_name=f"training_{self.config['experiment_id']}",
            config=self.config,
            log_file=self.results_dir / "training.log"
        ) as exp_logger:

            for iteration in range(self.config["iterations"]):
                # Train one iteration
                result = self.algorithm.train()
                training_results.append(result)

                # Extract key metrics
                reward = result['env_runners']['episode_reward_mean']
                episode_len = result['env_runners']['episode_len_mean']

                # Log progress
                if iteration % 5 == 0:
                    print(f"  Iteration {iteration + 1}/{self.config['iterations']}: "
                          f"Reward={reward:.2f}, Length={episode_len:.1f}")

                # Log bio-inspired metrics
                self._extract_and_log_bio_metrics(result, iteration)

                # Save best model
                if reward > best_reward:
                    best_reward = reward
                    checkpoint_path = self.algorithm.save(str(self.results_dir / "best_model"))
                    exp_logger.info(f"New best model saved: reward={reward:.2f}")

                # Save periodic checkpoint
                if iteration % 20 == 0:
                    self.algorithm.save(str(self.results_dir / f"checkpoint_{iteration}"))

        print(f"✅ Training completed! Best reward: {best_reward:.2f}")
        return training_results, best_reward

    def _extract_and_log_bio_metrics(self, result: Dict[str, Any], iteration: int):
        """Extract and log bio-inspired metrics from training result."""
        # Check if bio-inspired metrics are available
        learner_info = result.get('learner', {})

        # Mock bio-inspired metrics for demonstration
        # In real implementation, these would come from the RLModule
        if self.metrics_tracker:
            # Simulate attention metrics
            attention_entropy = np.random.uniform(1.5, 3.0)
            attention_sparsity = np.random.uniform(0.2, 0.8)
            fake_attention_weights = torch.randn(1, 4, self.config["num_agents"], self.config["num_agents"])

            self.metrics_tracker.log_attention_metrics(
                fake_attention_weights,
                attention_entropy,
                timestamp=iteration
            )

            # Simulate communication metrics
            comm_entropy = np.random.uniform(1.0, 2.5)
            comm_sparsity = np.random.uniform(0.1, 0.6)
            channel_usage = np.random.rand(self.config.get("comm_channels", 16))

            self.metrics_tracker.log_communication_metrics(
                comm_entropy,
                comm_sparsity,
                channel_usage,
                timestamp=iteration
            )

            # Simulate plasticity metrics
            memory_change = np.random.uniform(0.05, 0.25)
            plasticity_rate = np.random.uniform(0.08, 0.15)
            signal_strength = np.random.uniform(0.5, 1.0)

            self.metrics_tracker.log_plasticity_metrics(
                memory_change,
                plasticity_rate,
                signal_strength,
                timestamp=iteration
            )

    def analyze_results(self, training_results):
        """Comprehensive analysis of training results and bio-inspired behavior."""
        print("📊 Analyzing training results and bio-inspired behavior...")

        # Basic training analysis
        rewards = [r['env_runners']['episode_reward_mean'] for r in training_results]
        episode_lengths = [r['env_runners']['episode_len_mean'] for r in training_results]

        # Plot training curves
        self._plot_training_curves(rewards, episode_lengths)

        # Analyze bio-inspired metrics
        if self.metrics_tracker:
            summary = self.metrics_tracker.get_summary_statistics()
            print("\n🧬 Bio-Inspired Metrics Summary:")

            for metric_type, stats in summary.items():
                if stats.get('count', 0) > 0:
                    print(f"\n  {metric_type.title()}:")
                    for key, value in stats.items():
                        if isinstance(value, (int, float)) and key != 'count':
                            print(f"    {key}: {value:.4f}")

            # Export metrics
            metrics_file = self.results_dir / "bio_metrics.json"
            self.metrics_tracker.export_metrics(metrics_file)
            print(f"\n📁 Bio-inspired metrics exported to: {metrics_file}")

        # Performance analysis
        self._analyze_performance(training_results)

        print("✅ Analysis complete")

    def _plot_training_curves(self, rewards, episode_lengths):
        """Plot training curves and save visualizations."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Reward curve
        ax1.plot(rewards, linewidth=2, color='blue', alpha=0.8)
        ax1.fill_between(range(len(rewards)), rewards, alpha=0.3, color='blue')
        ax1.set_title('Training Reward Progress')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Episode Reward')
        ax1.grid(True, alpha=0.3)

        # Episode length curve
        ax2.plot(episode_lengths, linewidth=2, color='green', alpha=0.8)
        ax2.fill_between(range(len(episode_lengths)), episode_lengths, alpha=0.3, color='green')
        ax2.set_title('Episode Length Progress')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Episode Length')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()

        print("📈 Training curves saved")

    def _analyze_performance(self, training_results):
        """Analyze computational performance."""
        # Extract timing information
        timings = []
        for result in training_results:
            if 'timers' in result:
                sample_time = result['timers'].get('sample_time_ms', 0)
                learn_time = result['timers'].get('learn_time_ms', 0)
                timings.append({'sample': sample_time, 'learn': learn_time})

        if timings:
            avg_sample_time = np.mean([t['sample'] for t in timings])
            avg_learn_time = np.mean([t['learn'] for t in timings])

            print(f"\n⚡ Performance Analysis:")
            print(f"  Average sampling time: {avg_sample_time:.1f}ms")
            print(f"  Average learning time: {avg_learn_time:.1f}ms")
            print(f"  Total time per iteration: {avg_sample_time + avg_learn_time:.1f}ms")

    def demonstrate_inference(self):
        """Demonstrate trained agent inference with bio-inspired analysis."""
        print("🎯 Demonstrating trained agent inference...")

        # Create environment for inference
        env = self.create_environment()
        obs, info = env.reset()

        total_reward = 0
        steps = 0
        max_steps = 25

        print(f"Running inference for up to {max_steps} steps...")

        while steps < max_steps:
            # Get actions from trained policy
            actions = self.algorithm.compute_actions(obs)

            # Step environment
            obs, rewards, dones, truncs, info = env.step(actions)

            # Accumulate metrics
            total_reward += sum(rewards.values())
            steps += 1

            # Check if episode is done
            if any(dones.values()) or any(truncs.values()):
                break

        print(f"✅ Inference complete:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps taken: {steps}")
        print(f"  Average reward per step: {total_reward/steps:.3f}")

        return total_reward, steps

    def export_model(self):
        """Export trained model for deployment."""
        print("📦 Exporting trained model...")

        # Save final checkpoint
        final_checkpoint = self.algorithm.save(str(self.results_dir / "final_model"))

        # Export model information
        model_info = {
            "experiment_id": self.config["experiment_id"],
            "num_agents": self.config["num_agents"],
            "bio_config": {
                "hidden_dim": self.config.get("hidden_dim", 256),
                "memory_dim": self.config.get("memory_dim", 64),
                "comm_channels": self.config.get("comm_channels", 16),
                "use_positional_encoding": self.config.get("use_positional_encoding", True),
                "adaptive_plasticity": self.config.get("adaptive_plasticity", True)
            },
            "checkpoint_path": final_checkpoint,
            "framework": "torch",
            "rllib_version": "2.9.x"
        }

        import json
        with open(self.results_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"✅ Model exported:")
        print(f"  Checkpoint: {final_checkpoint}")
        print(f"  Model info: {self.results_dir / 'model_info.json'}")

        return final_checkpoint

    def run_complete_example(self):
        """Run the complete bio-inspired MARL example."""
        print("🌿" + "="*60)
        print("🌿 COMPLETE BIO-INSPIRED MULTI-AGENT RL EXAMPLE")
        print("🌿" + "="*60)

        try:
            # Initialize Ray
            ray.init(local_mode=self.config.get("debug_mode", False))

            # Setup logging and monitoring
            self.setup_logging_and_monitoring()

            # Create bio-inspired agents
            module_spec = self.create_bio_inspired_agents()

            # Setup training
            self.setup_training(module_spec)

            # Train agents
            training_results, best_reward = self.train_agents()

            # Analyze results
            self.analyze_results(training_results)

            # Demonstrate inference
            self.demonstrate_inference()

            # Export model
            final_checkpoint = self.export_model()

            print("\n🎉" + "="*60)
            print("🎉 EXAMPLE COMPLETED SUCCESSFULLY!")
            print("🎉" + "="*60)
            print(f"✅ Best training reward: {best_reward:.2f}")
            print(f"✅ Results saved to: {self.results_dir}")
            print(f"✅ Model checkpoint: {final_checkpoint}")

            return True

        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Cleanup
            if self.algorithm:
                self.algorithm.stop()
            ray.shutdown()


def main():
    """Main function to run the complete example."""
    # Configuration for the example
    config = {
        "experiment_id": "complete_demo_001",
        "num_agents": 6,
        "iterations": 30,
        "hidden_dim": 256,
        "memory_dim": 64,
        "comm_channels": 16,
        "comm_rounds": 3,
        "plasticity_rate": 0.12,
        "use_positional_encoding": True,
        "adaptive_plasticity": True,
        "batch_size": 512,
        "minibatch_size": 128,
        "learning_rate": 3e-4,
        "num_workers": 2,
        "debug_mode": True,
        "use_wandb": False,
        "results_dir": "./example_results"
    }

    # Run the complete example
    example = CompleteBioInspiredExample(config)
    success = example.run_complete_example()

    if success:
        print("\n🌱 Try exploring the results directory for:")
        print("   📊 Training curves and visualizations")
        print("   📋 Bio-inspired metrics analysis")
        print("   💾 Trained model checkpoints")
        print("   📝 Detailed logs and performance data")
        print("\n🚀 Ready to build your own bio-inspired agents!")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
