# File: marlcomm/run_optimized_experiment.py
"""
Optimized Integration Script for High-Performance MARL Training

This script integrates the hardware optimization module with the research
framework to maximize training speed on high-end systems.

Optimizations:
- GPU-accelerated neural networks
- Parallel environment simulation across CPU cores
- Large batch training for GPU efficiency
- Optimized Ray configuration
- Performance monitoring
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import ray
from ray import tune

# Import our modules
from hardware_optimization import HardwareOptimizer, OptimizedTrainingPipeline
from research_framework import ResearchHypothesis, ExperimentConfig, HypothesisGenerator
from emergence_environments import create_emergence_environment
from communication_metrics import CommunicationAnalyzer, CommunicationEvent
from reward_engineering import RewardEngineer, RewardContext, RewardPresets
from analysis_visualization import ExperimentData, ExperimentReporter
from rl_module import create_nature_comm_module_spec
from environment import register_nature_comm_environment


class OptimizedExperiment:
    """High-performance experiment runner."""
    
    def __init__(self, hypothesis: ResearchHypothesis, output_dir: Path):
        self.hypothesis = hypothesis
        self.output_dir = output_dir
        self.optimizer = HardwareOptimizer()
        self.pipeline = OptimizedTrainingPipeline(hypothesis.id)
        
        # Performance tracking
        self.performance_metrics = {
            'env_steps_per_second': [],
            'train_samples_per_second': [],
            'gpu_utilization': [],
            'cpu_utilization': []
        }
        
    def setup_optimized_environment(self):
        """Create environment with optimizations."""
        
        # Register environment creation function
        def create_env(env_config):
            # Extract config
            env_type = env_config.get("env_type", "foraging")
            n_agents = env_config.get("n_agents", 5)
            
            # Create base environment
            env = create_emergence_environment(
                env_type=env_type,
                n_agents=n_agents,
                grid_size=(30, 30),
                episode_length=200
            )
            
            # Wrap with reward engineering
            reward_engineer = self._create_reward_engineer(env_type)
            
            # Return wrapped environment
            from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
            return ParallelPettingZooEnv(env)
        
        # Register with Ray
        ray.tune.registry.register_env("optimized_emergence_env", create_env)
        
    def _create_reward_engineer(self, env_type: str) -> RewardEngineer:
        """Create appropriate reward engineer."""
        if "ant" in self.hypothesis.id:
            return RewardPresets.ant_colony_foraging()['engineer']
        elif "bee" in self.hypothesis.id:
            return RewardPresets.bee_waggle_dance()['engineer']
        else:
            engineer = RewardEngineer(env_type)
            engineer.create_implicit_reward_structure("multi_principle")
            return engineer
    
    def create_optimized_config(self):
        """Create training configuration optimized for hardware."""
        
        # Environment config
        env_config = {
            "env_type": self._get_env_type(),
            "n_agents": 5,
            "debug": False
        }
        
        # Get observation and action spaces
        test_env = create_emergence_environment(env_config["env_type"], n_agents=5)
        obs_space = test_env.observation_space("agent_0")
        act_space = test_env.action_space("agent_0")
        
        # Create RL module spec
        rl_module_spec = create_nature_comm_module_spec(
            obs_space, act_space,
            model_config={
                "comm_channels": 8,
                "memory_size": 32,  # Larger memory for GPU
                **self.optimizer.optimize_model_architecture(
                    obs_space.shape[0],
                    act_space.shape[0] if hasattr(act_space, 'shape') else act_space.n,
                    8  # communication channels
                )
            }
        )
        
        # Get optimized PPO config
        ppo_config = self.optimizer.get_optimized_ppo_config(
            "optimized_emergence_env",
            env_config,
            rl_module_spec
        )
        
        # Add callbacks for metrics
        ppo_config = ppo_config.callbacks(PerformanceTrackingCallbacks)
        
        return ppo_config
    
    def _get_env_type(self) -> str:
        """Map hypothesis to environment type."""
        if "foraging" in self.hypothesis.id:
            return "foraging"
        elif "predator" in self.hypothesis.id or "danger" in self.hypothesis.id:
            return "predator_prey"
        elif "synchronization" in self.hypothesis.id:
            return "temporal_coordination"
        else:
            return "foraging"
    
    def run_optimized_training(self, num_iterations: int = 200):
        """Run training with performance monitoring."""
        
        print(f"\n🚀 Starting optimized experiment: {self.hypothesis.id}")
        print(f"   Expected throughput: {self._estimate_throughput()} steps/second")
        
        # Setup
        self.setup_optimized_environment()
        config = self.create_optimized_config()
        
        # Initialize Ray with optimizations
        ray_config = self.optimizer.optimize_ray_init()
        if not ray.is_initialized():
            ray.init(**ray_config)
        
        # Create trainer
        algo = config.build()
        
        # Training loop with performance monitoring
        start_time = time.time()
        results_history = []
        
        for i in range(num_iterations):
            # Train
            result = algo.train()
            results_history.append(result)
            
            # Extract performance metrics
            if "sampler_perf" in result:
                perf = result["sampler_perf"]
                self.performance_metrics['env_steps_per_second'].append(
                    perf.get("mean_env_steps_per_sec", 0)
                )
            
            if "gpu_memory_used_mb" in result.get("custom_metrics", {}):
                self.performance_metrics['gpu_utilization'].append(
                    result["custom_metrics"]["gpu_memory_used_mb"]
                )
            
            # Progress report every 10 iterations
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                steps = result.get("num_env_steps_sampled", 0)
                throughput = steps / elapsed
                
                print(f"\nIteration {i+1}/{num_iterations}")
                print(f"  Episode Reward: {result.get('env_runners', {}).get('episode_reward_mean', 0):.2f}")
                print(f"  Total Steps: {steps:,}")
                print(f"  Throughput: {throughput:.0f} steps/s")
                print(f"  Time Elapsed: {elapsed/60:.1f} minutes")
                
                if self.optimizer.profile.gpu_available:
                    gpu_mb = torch.cuda.memory_allocated() / 1e6
                    print(f"  GPU Memory: {gpu_mb:.0f} MB")
        
        # Save final checkpoint
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        algo.save(str(checkpoint_dir))
        
        # Analyze results
        total_time = time.time() - start_time
        total_steps = results_history[-1].get("num_env_steps_sampled", 0)
        
        print(f"\n✅ Training Complete!")
        print(f"   Total Time: {total_time/60:.1f} minutes")
        print(f"   Total Steps: {total_steps:,}")
        print(f"   Average Throughput: {total_steps/total_time:.0f} steps/s")
        
        return algo, results_history
    
    def _estimate_throughput(self) -> int:
        """Estimate expected training throughput."""
        total_envs = (self.optimizer.profile.optimal_num_workers * 
                     self.optimizer.profile.optimal_num_envs_per_worker)
        
        if self.optimizer.profile.gpu_available:
            return total_envs * 50  # 50 Hz per environment with GPU
        else:
            return total_envs * 30  # 30 Hz per environment CPU only
    
    def analyze_performance(self, results_history):
        """Analyze training performance."""
        
        # Create performance report
        perf_report = {
            "hardware": {
                "gpu": self.optimizer.profile.gpu_name,
                "cpu": self.optimizer.profile.cpu_name,
                "ram": f"{self.optimizer.profile.ram_total_gb:.0f} GB"
            },
            "configuration": {
                "workers": self.optimizer.profile.optimal_num_workers,
                "envs_per_worker": self.optimizer.profile.optimal_num_envs_per_worker,
                "batch_size": self.optimizer.profile.optimal_batch_size
            },
            "performance": {
                "avg_throughput": np.mean(self.performance_metrics['env_steps_per_second']),
                "peak_throughput": np.max(self.performance_metrics['env_steps_per_second']),
                "gpu_memory_peak": np.max(self.performance_metrics['gpu_utilization']) if self.performance_metrics['gpu_utilization'] else 0
            }
        }
        
        return perf_report


class PerformanceTrackingCallbacks:
    """Callbacks for detailed performance monitoring."""
    
    @staticmethod
    def on_train_result(*, algorithm, result, **kwargs):
        """Track performance metrics."""
        
        # GPU metrics
        if torch.cuda.is_available():
            result["custom_metrics"]["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / 1e6
            result["custom_metrics"]["gpu_memory_cached_mb"] = torch.cuda.memory_reserved() / 1e6
            
            # Try to get GPU utilization (requires nvidia-ml-py)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    result["custom_metrics"]["gpu_utilization_percent"] = gpus[0].load * 100
                    result["custom_metrics"]["gpu_temperature"] = gpus[0].temperature
            except:
                pass
        
        # CPU metrics
        try:
            import psutil
            result["custom_metrics"]["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            result["custom_metrics"]["memory_percent"] = psutil.virtual_memory().percent
        except:
            pass
        
        # Training efficiency
        if "num_env_steps_sampled" in result and "time_this_iter_s" in result:
            result["custom_metrics"]["steps_per_second_this_iter"] = (
                result["num_env_steps_sampled_this_iter"] / result["time_this_iter_s"]
            )


def compare_optimized_vs_baseline():
    """Run comparison between optimized and baseline configurations."""
    
    print("📊 Performance Comparison: Optimized vs Baseline")
    print("=" * 60)
    
    # Create test hypothesis
    hypothesis = HypothesisGenerator.generate_ant_colony_hypotheses()[0]
    
    # Baseline configuration (from original code)
    print("\n1️⃣ Baseline Configuration:")
    print("   Workers: 4")
    print("   Envs per worker: 1")
    print("   Batch size: 1024")
    print("   Total parallel envs: 4")
    
    # Optimized configuration
    optimizer = HardwareOptimizer()
    print(f"\n2️⃣ Optimized Configuration:")
    print(f"   Workers: {optimizer.profile.optimal_num_workers}")
    print(f"   Envs per worker: {optimizer.profile.optimal_num_envs_per_worker}")
    print(f"   Batch size: {optimizer.profile.optimal_batch_size}")
    print(f"   Total parallel envs: {optimizer.profile.optimal_num_workers * optimizer.profile.optimal_num_envs_per_worker}")
    
    # Calculate speedup
    baseline_throughput = 4 * 20  # 4 envs at ~20Hz
    optimized_throughput = optimizer.profile.optimal_num_workers * optimizer.profile.optimal_num_envs_per_worker * 50
    
    speedup = optimized_throughput / baseline_throughput
    
    print(f"\n📈 Expected Performance Improvement:")
    print(f"   Throughput speedup: {speedup:.1f}x")
    print(f"   Time to 1M steps:")
    print(f"     - Baseline: {1_000_000 / baseline_throughput / 60:.1f} minutes")
    print(f"     - Optimized: {1_000_000 / optimized_throughput / 60:.1f} minutes")
    
    # Memory usage comparison
    print(f"\n💾 Memory Efficiency:")
    print(f"   Baseline uses: ~4GB RAM")
    print(f"   Optimized uses: ~{optimizer.profile.optimal_num_workers * 2}GB RAM")
    print(f"   Available: {optimizer.profile.ram_total_gb:.0f}GB RAM")


def main():
    """Main entry point for optimized experiments."""
    
    parser = argparse.ArgumentParser(
        description="Run optimized MARL communication experiments"
    )
    
    parser.add_argument(
        "--experiment",
        choices=["ant_foraging", "benchmark", "full_campaign"],
        default="ant_foraging",
        help="Experiment to run"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("optimized_results"),
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    if args.experiment == "benchmark":
        # Run performance comparison
        compare_optimized_vs_baseline()
        
    elif args.experiment == "ant_foraging":
        # Run single optimized experiment
        hypothesis = HypothesisGenerator.generate_ant_colony_hypotheses()[0]
        
        output_dir = args.output_dir / hypothesis.id / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        experiment = OptimizedExperiment(hypothesis, output_dir)
        algo, results = experiment.run_optimized_training(args.iterations)
        
        # Analyze performance
        perf_report = experiment.analyze_performance(results)
        
        print("\n📊 Performance Report:")
        print(f"   Average throughput: {perf_report['performance']['avg_throughput']:.0f} steps/s")
        print(f"   Peak throughput: {perf_report['performance']['peak_throughput']:.0f} steps/s")
        
    elif args.experiment == "full_campaign":
        # Run full research campaign with optimizations
        print("🚀 Running full optimized research campaign...")
        print("   This will utilize all available hardware for maximum speed")
        
        # TODO: Implement full campaign runner with optimizations
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
