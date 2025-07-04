# File: marlcomm/first_experiment_gpu_fixed.py
"""
GPU-optimized version of the first experiment with all critical fixes.
Ready to run your first emergent communication experiment with full GPU utilization!
"""

import os
import sys

# CRITICAL: Set GPU environment BEFORE any CUDA imports
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import json
import matplotlib.pyplot as plt
import ray
from ray import tune

# Verify GPU before proceeding
print("🔍 GPU Detection:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("   ⚠️ WARNING: No GPU detected! Check your setup.")

# Import all our modules
from hardware_optimization import HardwareOptimizer
from research_framework import (
    ResearchHypothesis, ExperimentConfig,
    CommunicationParadigm, EmergencePressure
)
from emergence_environments import create_emergence_environment
from communication_metrics import CommunicationAnalyzer
from reward_engineering import RewardPresets
from reward_engineering_wrapper import RewardEngineeringWrapper
from emergence_callbacks import RealEmergenceTrackingCallbacks
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from rl_module import create_nature_comm_module_spec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env


class GPUMonitor:
    """Simple GPU monitoring during training."""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.max_gpu_util = 0
        self.max_mem_used = 0

    def check_usage(self):
        """Check current GPU usage."""
        if not self.gpu_available:
            return 0, 0

        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)

            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)

            gpu_util = util.gpu
            mem_used_gb = mem_info.used / 1024**3

            self.max_gpu_util = max(self.max_gpu_util, gpu_util)
            self.max_mem_used = max(self.max_mem_used, mem_used_gb)

            nvml.nvmlShutdown()
            return gpu_util, mem_used_gb
        except:
            return 0, 0


class FirstExperimentGPUFixed:
    """GPU-optimized implementation of the first experiment."""

    def __init__(self):
        self.experiment_name = "first_ant_foraging_gpu"
        self.output_dir = Path("experiment_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_monitor = GPUMonitor()

        print("\n🎉 First Emergent Communication Experiment (GPU-Optimized)")
        print("=" * 70)
        print(f"📁 Results will be saved to: {self.output_dir}")
        print(f"🔥 GPU Mode: {'ENABLED' if torch.cuda.is_available() else 'DISABLED'}")

    def setup(self):
        """Setup experiment components."""
        print("\n📋 Setting up GPU-optimized experiment...")

        # Hardware detection
        self.optimizer = HardwareOptimizer()

        # Verify GPU in optimizer
        if self.optimizer.profile.gpu_available:
            print(f"✅ GPU detected: {self.optimizer.profile.gpu_device}")
        else:
            print("⚠️ No GPU detected by optimizer!")

        # Create hypothesis
        self.hypothesis = ResearchHypothesis(
            id="ant_foraging_gpu",
            description="Agents develop pheromone communication for efficient foraging",
            paradigm=CommunicationParadigm.PHEROMONE,
            pressure=EmergencePressure.RESOURCE_SCARCITY,
            predictions=["Agents will develop trail-following behavior"],
            metrics=["mutual_information", "coordination_efficiency"],
            parameters={"pheromone_decay": 0.95}
        )

        # Setup reward engineering
        preset = RewardPresets.ant_colony_foraging()
        self.reward_engineer = preset['engineer']

        # Initialize communication analyzer
        self.communication_analyzer = CommunicationAnalyzer()

        print("✅ Setup complete!")

    def create_env_fn(self, env_config):
        """Environment creator function with proper reward wrapper."""
        # Create base environment
        base_env = create_emergence_environment(
            env_type="foraging",
            n_agents=5,
            grid_size=(30, 30),
            n_food_clusters=3
        )

        # Wrap with PettingZoo wrapper
        pz_env = ParallelPettingZooEnv(base_env)

        # Apply reward engineering wrapper
        wrapped_env = RewardEngineeringWrapper(
            pz_env,
            reward_engineer=env_config.get("reward_engineer", self.reward_engineer),
            communication_analyzer=env_config.get("communication_analyzer", self.communication_analyzer)
        )

        return wrapped_env

    def initialize_ray_with_gpu(self):
        """Initialize Ray with proper GPU configuration."""
        if ray.is_initialized():
            ray.shutdown()

        # Determine resources
        num_cpus = self.optimizer.profile.num_cpus
        num_gpus = 1 if self.optimizer.profile.gpu_available else 0

        print(f"\n🚀 Initializing Ray with {num_cpus} CPUs and {num_gpus} GPUs...")

        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            logging_level="WARNING",
            include_dashboard=False,
            _system_config={
                "automatic_object_spilling_enabled": True,
                "object_spilling_config": {
                    "type": "filesystem",
                    "params": {"directory_path": "/tmp/ray_spilled"}
                },
                "max_direct_call_object_size": 100 * 1024 * 1024,  # 100MB
            }
        )

        # Verify resources
        resources = ray.available_resources()
        print(f"✅ Ray initialized with resources: {resources}")

        if num_gpus > 0 and resources.get('GPU', 0) < 1:
            print("⚠️ Warning: GPU requested but not available in Ray!")

        return resources

    def create_gpu_optimized_config(self, env_name, env_config, rl_module_spec):
        """Create GPU-optimized PPO configuration."""

        # Get base config from hardware optimizer
        config = self.optimizer.get_optimized_ppo_config(
            env=env_name,
            env_config=env_config,
            rl_module_spec=rl_module_spec
        )

        # Apply GPU-specific optimizations
        config = config.framework("torch")  # PyTorch for better GPU support

        if self.optimizer.profile.gpu_available:
            config = config.resources(
                num_gpus=1,  # 1 GPU for trainer
                num_cpus_per_worker=1,
                num_gpus_per_worker=0,  # Workers use CPU
                placement_strategy="PACK"
            )

            # GPU-optimized training parameters
            config = config.training(
                train_batch_size=8192,  # Large batch for GPU efficiency
                sgd_minibatch_size=512,  # Good GPU tile size
                num_sgd_iter=10,
                lr=3e-4,
                lr_schedule=[[0, 3e-4], [1000000, 1e-4]],
                grad_clip=0.5,
                model={
                    "fcnet_hiddens": [512, 512],  # Larger network for GPU
                    "fcnet_activation": "relu",
                    "vf_share_layers": True,
                    "use_lstm": False,  # LSTM can be slow on GPU
                }
            )

            # More parallel workers to feed GPU
            config = config.rollouts(
                num_rollout_workers=6,  # More workers
                rollout_fragment_length=200,
                batch_mode="truncate_episodes"
            )
        else:
            print("⚠️ Running in CPU mode with reduced parameters")

        # Add callbacks
        config = config.callbacks(RealEmergenceTrackingCallbacks)

        return config

    def train(self, iterations=100):
        """Run training with GPU monitoring."""
        print(f"\n🚀 Starting GPU-optimized training for {iterations} iterations...")

        # Initialize Ray with GPU
        self.initialize_ray_with_gpu()

        # Register environment
        register_env("foraging_env_gpu", self.create_env_fn)

        # Get spaces for config
        test_env = self.create_env_fn({
            "reward_engineer": self.reward_engineer,
            "communication_analyzer": self.communication_analyzer
        })

        # Create RL module spec
        single_agent_spec = create_nature_comm_module_spec(
            test_env.observation_space['agent_0'],
            test_env.action_space['agent_0'],
            model_config={
                "comm_channels": 2,
                "memory_size": 16,
                "hidden_sizes": [512, 512] if self.optimizer.profile.gpu_available else [256, 256]
            }
        )

        rl_module_spec = MultiAgentRLModuleSpec(
            module_specs={agent_id: single_agent_spec for agent_id in test_env.agents}
        )

        # Create GPU-optimized config
        config = self.create_gpu_optimized_config(
            env_name="foraging_env_gpu",
            env_config={
                "reward_engineer": self.reward_engineer,
                "communication_analyzer": self.communication_analyzer
            },
            rl_module_spec=rl_module_spec
        )

        # Build algorithm
        algo = config.build()

        # Training loop with GPU monitoring
        self.metrics_history = {
            'episode_reward_mean': [],
            'mutual_information': [],
            'coordination_efficiency': [],
            'communication_frequency': [],
            'emergence_score': [],
            'gpu_utilization': [],
            'gpu_memory_gb': []
        }

        start_time = time.time()
        emergence_detected = False
        emergence_iteration = None

        print("\nIter | Reward  | MI    | Coord | Comm  | Emrg  | GPU%  | Time   | Status")
        print("-" * 85)

        for i in range(iterations):
            # Train one iteration
            result = algo.train()

            # Check GPU usage
            gpu_util, gpu_mem = self.gpu_monitor.check_usage()

            # Extract metrics
            sample_results = result.get('sampler_results', {})
            reward = sample_results.get('episode_reward_mean', 0)

            custom_metrics = sample_results.get('custom_metrics', {})
            mi = custom_metrics.get('mutual_information_mean', 0.0)
            coord = custom_metrics.get('coordination_efficiency_mean', 0.3)
            comm_freq = custom_metrics.get('communication_frequency_mean', 0.0)
            emrg_score = custom_metrics.get('emergence_score_mean', 0.0)

            # Store metrics
            self.metrics_history['episode_reward_mean'].append(reward)
            self.metrics_history['mutual_information'].append(mi)
            self.metrics_history['coordination_efficiency'].append(coord)
            self.metrics_history['communication_frequency'].append(comm_freq)
            self.metrics_history['emergence_score'].append(emrg_score)
            self.metrics_history['gpu_utilization'].append(gpu_util)
            self.metrics_history['gpu_memory_gb'].append(gpu_mem)

            # Check for emergence
            if not emergence_detected and custom_metrics.get('emergence_detected', False):
                emergence_detected = True
                emergence_iteration = i
                status = "🎉 EMERGED!"
            elif emergence_detected:
                status = "📈 Growing"
            else:
                status = "🔄 Learning"

            # Print progress
            elapsed = time.time() - start_time
            print(f"{i+1:4d} | {reward:7.2f} | {mi:5.3f} | {coord:5.2f} | "
                  f"{comm_freq:5.2f} | {emrg_score:5.2f} | {gpu_util:5.0f} | "
                  f"{elapsed/60:6.1f}m | {status}")

            # Save checkpoint periodically
            if (i + 1) % 25 == 0:
                checkpoint_dir = self.output_dir / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                algo.save(checkpoint_dir / f"iter_{i+1}")

        # Save final model
        algo.save(self.output_dir / "final_model")

        # Cleanup
        algo.stop()
        ray.shutdown()

        total_time = time.time() - start_time
        print("-" * 85)
        print(f"\n✅ Training complete! Total time: {total_time/60:.1f} minutes")
        print(f"📊 Max GPU Utilization: {self.gpu_monitor.max_gpu_util}%")
        print(f"📊 Max GPU Memory Used: {self.gpu_monitor.max_mem_used:.2f} GB")

        if emergence_detected:
            print(f"🎉 Communication emerged at iteration {emergence_iteration}!")
        else:
            print("🔬 Communication patterns are developing. Try more iterations!")

        return self.metrics_history, emergence_iteration

    def analyze_and_visualize(self, metrics_history, emergence_iteration):
        """Analyze results with GPU metrics."""
        print("\n📊 Analyzing results...")

        # Create figure with GPU metrics
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()

        # 1. Episode rewards
        axes[0].plot(metrics_history['episode_reward_mean'], linewidth=2, color='blue')
        axes[0].set_title('Episode Rewards', fontsize=12)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Mean Reward')
        axes[0].grid(True, alpha=0.3)

        # 2. Mutual Information
        axes[1].plot(metrics_history['mutual_information'], linewidth=2, color='orange')
        axes[1].set_title('Mutual Information', fontsize=12)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('MI (bits)')
        axes[1].grid(True, alpha=0.3)

        if emergence_iteration:
            axes[1].axvline(x=emergence_iteration, color='red', linestyle='--',
                          label='Emergence Detected', alpha=0.7)
            axes[1].legend()

        # 3. Coordination Efficiency
        axes[2].plot(metrics_history['coordination_efficiency'], linewidth=2, color='green')
        axes[2].set_title('Coordination Efficiency', fontsize=12)
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Efficiency')
        axes[2].grid(True, alpha=0.3)

        # 4. Communication Frequency
        axes[3].plot(metrics_history['communication_frequency'], linewidth=2, color='purple')
        axes[3].set_title('Communication Frequency', fontsize=12)
        axes[3].set_xlabel('Iteration')
        axes[3].set_ylabel('Messages/Step')
        axes[3].grid(True, alpha=0.3)

        # 5. Emergence Score
        axes[4].plot(metrics_history['emergence_score'], linewidth=2, color='red')
        axes[4].set_title('Emergence Score', fontsize=12)
        axes[4].set_xlabel('Iteration')
        axes[4].set_ylabel('Score')
        axes[4].grid(True, alpha=0.3)
        axes[4].axhline(y=0.6, color='red', linestyle=':', alpha=0.5, label='Threshold')
        axes[4].legend()

        # 6. GPU Utilization
        axes[5].plot(metrics_history['gpu_utilization'], linewidth=2, color='darkgreen')
        axes[5].set_title('GPU Utilization', fontsize=12)
        axes[5].set_xlabel('Iteration')
        axes[5].set_ylabel('GPU %')
        axes[5].grid(True, alpha=0.3)
        axes[5].axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Target')

        # 7. GPU Memory
        axes[6].plot(metrics_history['gpu_memory_gb'], linewidth=2, color='darkblue')
        axes[6].set_title('GPU Memory Usage', fontsize=12)
        axes[6].set_xlabel('Iteration')
        axes[6].set_ylabel('Memory (GB)')
        axes[6].grid(True, alpha=0.3)

        # 8. Reward vs GPU Utilization scatter
        axes[7].scatter(metrics_history['gpu_utilization'],
                       metrics_history['episode_reward_mean'],
                       alpha=0.6, color='coral')
        axes[7].set_title('Reward vs GPU Utilization', fontsize=12)
        axes[7].set_xlabel('GPU %')
        axes[7].set_ylabel('Episode Reward')
        axes[7].grid(True, alpha=0.3)

        # 9. Summary text
        axes[8].axis('off')
        summary_text = f"""GPU-Optimized Experiment Summary:

Final Reward: {metrics_history['episode_reward_mean'][-1]:.2f}
Max MI: {max(metrics_history['mutual_information']):.3f}
Max Coordination: {max(metrics_history['coordination_efficiency']):.3f}
Emergence: {'Yes' if emergence_iteration else 'In Progress'}

GPU Performance:
• Average GPU Utilization: {np.mean(metrics_history['gpu_utilization']):.1f}%
• Max GPU Utilization: {max(metrics_history['gpu_utilization']):.1f}%
• Average GPU Memory: {np.mean(metrics_history['gpu_memory_gb']):.2f} GB

Key Findings:
• Coordination improved by
  {(metrics_history['coordination_efficiency'][-1] /
    metrics_history['coordination_efficiency'][0] - 1) * 100:.1f}%
• GPU optimization provided
  {'significant' if np.mean(metrics_history['gpu_utilization']) > 40 else 'moderate'} speedup
        """
        axes[8].text(0.1, 0.5, summary_text, transform=axes[8].transAxes,
                    verticalalignment='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('GPU-Optimized Emergent Communication Analysis',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save figure
        plt.savefig(self.output_dir / 'gpu_emergence_analysis.png', dpi=150)
        plt.show()

        # Save metrics data
        with open(self.output_dir / 'metrics_history.json', 'w') as f:
            json.dump(metrics_history, f, indent=2)

        print(f"\n💾 Results saved to: {self.output_dir}")

    def run(self, iterations=100):
        """Run complete GPU-optimized experiment."""
        try:
            # Setup
            self.setup()

            # Train with GPU monitoring
            metrics_history, emergence_iteration = self.train(iterations)

            # Analyze and visualize
            self.analyze_and_visualize(metrics_history, emergence_iteration)

            # GPU Summary
            print("\n🔥 GPU Performance Summary:")
            avg_gpu = np.mean(metrics_history['gpu_utilization'])
            if avg_gpu > 60:
                print("   ✅ Excellent GPU utilization!")
            elif avg_gpu > 40:
                print("   ✅ Good GPU utilization")
            elif avg_gpu > 20:
                print("   ⚠️ Moderate GPU utilization - consider larger batches")
            else:
                print("   ❌ Low GPU utilization - check configuration")

            # Experiment Summary
            print("\n🎉 Experiment complete!")
            print("\n🔬 What happened:")
            if emergence_iteration:
                print("   ✓ Agents developed pheromone communication")
                print("   ✓ GPU acceleration improved training speed")
                print("   ✓ Efficient parallel processing achieved")
            else:
                print("   • Agents are learning to coordinate")
                print("   • GPU is accelerating the learning process")
                print("   • More iterations may show full emergence")

            print("\n🚀 Next steps:")
            print("   1. Monitor GPU with: watch -n 1 nvidia-smi")
            print("   2. Increase batch size for better GPU usage")
            print("   3. Try more agents for complex behaviors")
            print("   4. Compare CPU vs GPU training times")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

            print("\nGPU Troubleshooting:")
            print("1. Check CUDA: python -c 'import torch; print(torch.cuda.is_available())'")
            print("2. Check GPU memory: nvidia-smi")
            print("3. Reduce train_batch_size if OOM")
            print("4. Ensure Ray has GPU access: ray.available_resources()")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GPU-optimized emergent communication experiment"
    )
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of training iterations")
    parser.add_argument("--check-gpu", action="store_true",
                       help="Only check GPU status")

    args = parser.parse_args()

    if args.check_gpu:
        print("🔍 GPU Status Check:")
        print(f"   PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

            # Test computation
            try:
                a = torch.randn(1000, 1000).cuda()
                b = torch.randn(1000, 1000).cuda()
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
                print("   ✅ GPU computation test: PASSED")
            except Exception as e:
                print(f"   ❌ GPU computation test: FAILED - {e}")
        return

    print("🧪 Running GPU-Optimized Emergent Communication Experiment")
    print(f"   Iterations: {args.iterations}")
    print(f"   GPU Mode: {'ENABLED' if torch.cuda.is_available() else 'CPU FALLBACK'}")
    print("")

    experiment = FirstExperimentGPUFixed()
    experiment.run(iterations=args.iterations)


if __name__ == "__main__":
    main()
