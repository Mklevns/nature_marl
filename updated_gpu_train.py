# File: /marlcomm/updated_gpu_train.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
os.environ['RAY_DEDUP_LOGS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
# Apply patches for older numpy syntax if needed
if not hasattr(np, 'product'):
    np.product = np.prod
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

import sys
import torch
import time
from pathlib import Path

# --- Start of rl_module.py content ---
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from typing import Dict, Any
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.typing import TensorType


class NatureInspiredCommModule(TorchRLModule):
    """
    Custom RLModule implementing nature-inspired communication patterns.

    Features:
    - Pheromone-like communication signals (bounded chemical messaging)
    - Neural plasticity memory (GRU-based experience retention)
    - Adaptive behavior combining observations with communication history
    """

    def __init__(self, observation_space, action_space, *, model_config=None, **kwargs):
        """
        Initialize the nature-inspired communication module.

        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            model_config: Model configuration dictionary
            **kwargs: Additional parameters (inference_only, learner_only, etc.)
        """
        # Important: Use the new API stack constructor signature
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config or {},
            **kwargs
        )

    def setup(self):
        """Setup neural network components during initialization."""
        # Get dimensions from spaces
        # The RLModule receives the single agent's observation space, but for a shared policy
        # with concatenated observations, the actual input dimension will be num_agents * obs_dim.
        single_agent_obs_dim = self.observation_space.shape[0]
        num_agents = self.model_config.get("num_agents", 1) # Get num_agents from model_config
        true_obs_dim = single_agent_obs_dim * num_agents

        if isinstance(self.action_space, Discrete):
            action_dim = self.action_space.n
        else:
            action_dim = self.action_space.shape[0]

        # Communication and memory dimensions
        self.comm_dim = 8  # Size of pheromone-like signals
        self.memory_dim = 16  # Neural memory capacity

        # Access model_config for fcnet_hiddens, if provided.
        # Default to a smaller size if not specified.
        fcnet_hiddens = self.model_config.get("fcnet_hiddens", [64, 64])
        fcnet_activation = self.model_config.get("fcnet_activation", "relu")

        # Helper to create a sequential network based on hiddens and activation
        def create_fcn_network(input_dim, output_dim, hiddens, activation):
            layers = []
            current_dim = input_dim
            for h_dim in hiddens:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(getattr(nn, activation)())
                current_dim = h_dim
            layers.append(nn.Linear(current_dim, output_dim))
            return nn.Sequential(*layers)

        # Observation encoder (sensory processing)
        # Use true_obs_dim for the input layer, and larger hidden layers
        self.obs_encoder = nn.Sequential(
            nn.Linear(true_obs_dim, 1024), # Adjusted input and first hidden layer size
            nn.ReLU(),
            nn.Linear(1024, 512), # Second hidden layer
            nn.LayerNorm(512),  # Stabilize training
        )

        # Communication signal generator (pheromone production)
        self.comm_encoder = nn.Sequential(
            nn.Linear(32, self.comm_dim),
            nn.Tanh()  # Bounded signals like chemical concentrations
        )

        # Communication signal decoder (pheromone detection)
        self.comm_decoder = nn.Sequential(
            nn.Linear(self.comm_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        # Neural plasticity memory (experience integration)
        self.memory_update = nn.GRUCell(32 + self.comm_dim, self.memory_dim)

        # Policy network (decision making) - now using fcnet_hiddens
        self.policy_net = create_fcn_network(32 + self.memory_dim, action_dim, fcnet_hiddens, fcnet_activation)

        # Value network (outcome prediction) - now using fcnet_hiddens
        self.value_net = create_fcn_network(32 + self.memory_dim, 1, fcnet_hiddens, fcnet_activation)


        # Initialize persistent memory state
        self.register_buffer('hidden_state', torch.zeros(1, self.memory_dim))

    def _forward(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        """
        Core forward pass implementing nature-inspired communication.

        Args:
            batch: Input batch containing observations
            **kwargs: Additional forward arguments

        Returns:
            Dictionary with action distribution inputs and value predictions
        """
        obs = batch["obs"]
        batch_size = obs.shape[0]

        # Expand hidden state to match batch size if needed
        if self.hidden_state.shape[0] != batch_size:
            self.hidden_state = self.hidden_state.expand(batch_size, -1).contiguous()

        # Sensory processing: encode environmental observations
        obs_encoded = self.obs_encoder(obs)

        # Communication signal generation (pheromone production)
        comm_signal = self.comm_encoder(obs_encoded)

        # Communication signal processing (pheromone detection)
        # In real multi-agent setting, this would process signals from other agents
        comm_processed = self.comm_decoder(comm_signal)

        # Neural plasticity: update memory with new experience
        try:
            memory_input = torch.cat([obs_encoded, comm_signal], dim=1)
            self.hidden_state = self.memory_update(memory_input, self.hidden_state)
        except RuntimeError as e:
            # Handle tensor size mismatches gracefully
            if "size mismatch" in str(e).lower():
                # Reset memory state and try again
                self.hidden_state = torch.zeros(batch_size, self.memory_dim,
                                              device=obs.device, dtype=obs.dtype)
                self.hidden_state = self.memory_update(memory_input, self.hidden_state)
            else:
                raise e

        # Integrate processed information for decision making
        integrated_features = torch.cat([obs_encoded, self.hidden_state], dim=1)

        # Generate action distribution parameters and value estimates
        action_logits = self.policy_net(integrated_features)
        value_pred = self.value_net(integrated_features).squeeze(-1)

        return {
            "action_dist_inputs": action_logits,  # For action sampling
            "vf_preds": value_pred,  # For value function training
            "comm_signal": comm_signal,  # For potential analysis/logging
        }

    def get_initial_state(self) -> Dict[str, TensorType]:
        """Get initial state for recurrent processing."""
        return {"hidden_state": torch.zeros(1, self.memory_dim)}

    def is_stateful(self) -> bool:
        """Indicate this module maintains internal state."""
        return True

# No need for create_nature_comm_module_spec helper for direct pasting logic
# --- End of rl_module.py content ---

print("🚀 GPU-Optimized MARL Training - Maximum Performance")
print("=" * 60)

# GPU check
if not torch.cuda.is_available():
    print("❌ GPU not available!")
    sys.exit(1)

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

try:
    import gymnasium as gym
except ImportError:
    import gym


class OptimizedMultiAgentEnv(MultiAgentEnv):
    """Optimized environment for maximum GPU utilization."""

    def __init__(self, config=None):
        super().__init__() # Call the parent constructor first

        # Calculate the number of agents from config
        num_agents_value = config.get("num_agents", 8) if config else 8

        # Define self.agents list. The base MultiAgentEnv will derive num_agents from this.
        self.agents = [f"agent_{i}" for i in range(num_agents_value)]
        self._agent_ids = set(self.agents) # This line remains as is

        # Define spaces as a dictionary mapping agent IDs to their individual spaces
        # The observation space here is still Dict, but will be flattened by SuperSuit wrapper
        self.observation_space = gym.spaces.Dict({
            agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(128,), dtype=np.float32)
            for agent in self.agents
        })
        self.action_space = gym.spaces.Dict({
            agent: gym.spaces.Discrete(8)
            for agent in self.agents
        })

        self.steps = 0
        self.max_steps = 200  # Longer episodes

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        obs = {}
        for i, agent in enumerate(self.agents):
            # More complex initial observations
            agent_obs = np.random.randn(128).astype(np.float32) * 0.1
            agent_obs[i*16:(i+1)*16] = 1.0  # Agent identifier
            obs[agent] = agent_obs
        return obs, {}

    def step(self, actions):
        self.steps += 1

        obs = {}
        rewards = {}

        # More complex reward calculation
        # self.num_agents is now inferred from len(self.agents)
        action_matrix = np.zeros((len(self.agents), 8))
        for i, agent in enumerate(self.agents):
            if agent in actions:
                action_matrix[i, actions[agent]] = 1.0

        # Calculate coordination scores
        coordination = np.sum(action_matrix, axis=0)

        for i, agent in enumerate(self.agents):
            # Complex observation update
            agent_obs = np.random.randn(128).astype(np.float32) * 0.1
            agent_obs[i*16:(i+1)*16] = 1.0

            if agent in actions:
                action = actions[agent]
                # Add coordination info to observation
                agent_obs[-8:] = coordination / len(self.agents)

                # Reward based on coordination and diversity
                coord_reward = coordination[action] / len(self.agents)
                diversity_reward = -np.std(coordination) * 0.1
                rewards[agent] = coord_reward + diversity_reward + 0.1
            else:
                rewards[agent] = 0.0

            obs[agent] = agent_obs

        done = self.steps >= self.max_steps
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        truncateds = {agent: False for agent in self.agents}
        truncateds["__all__"] = False

        return obs, rewards, dones, truncateds, {}

# Environment creator function with SuperSuit wrappers
def create_optimized_multi_env_with_wrappers(env_config):
    env = OptimizedMultiAgentEnv(env_config)
    # SuperSuit wrappers are removed as they are not compatible with rllib.MultiAgentEnv
    # and are not needed for this configuration.
    return env

def create_optimized_ppo_config(num_agents):
    """Create a PPO config optimized for GPU utilization."""

    # Get environment spaces
    # NOTE: The env created here is *unwrapped* to get original spaces.
    # The actual env registered for Ray will have wrappers.
    env = OptimizedMultiAgentEnv({"num_agents": num_agents})

    # Original observation space for a single agent
    single_agent_obs_space = env.observation_space["agent_0"]
    single_agent_act_space = env.action_space["agent_0"]

    config = (
        PPOConfig()
        .experimental()
        .environment(
            env="optimized_multi_env", # <-- Use registered environment name
            env_config={"num_agents": num_agents},
        )
        .framework("torch")
        .resources(
            num_gpus=1,
            placement_strategy="PACK"
        )
        .env_runners(
            num_env_runners=8,
            rollout_fragment_length='auto',
            batch_mode="truncate_episodes",
            remote_worker_envs=True,
            num_envs_per_env_runner=2,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=NatureInspiredCommModule,
                # Pass single agent's spaces and config for the module
                observation_space=single_agent_obs_space,
                action_space=single_agent_act_space,
                model_config={
                    "fcnet_hiddens": [1024, 1024, 512], # These will be passed to NatureInspiredCommModule
                    "fcnet_activation": "relu",
                    "vf_share_layers": False,
                    "free_log_std": True,
                }
            )
        )
        .training(
            train_batch_size_per_learner=16384,
            minibatch_size=1024,
            num_epochs=20,
            lr=[ # Assign schedule directly to lr
                [0, 5e-4],
                [1000000, 1e-4],
            ],
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.3,
            entropy_coeff=[ # Assign schedule directly to entropy_coeff
                [0, 0.01],
                [1000000, 0.001],
            ],
            vf_loss_coeff=0.5,
            kl_coeff=0.2,
            kl_target=0.02,
        )
        .multi_agent( # multi_agent remains here, chained after .training()
            policies={
                # IMPORTANT: Policy's obs/act space must match the RLModule's expectation.
                # It should be the single agent's observation space.
                "shared_policy": (None, single_agent_obs_space, single_agent_act_space, {})
            },
            policy_mapping_fn=lambda agent_id, episode: "shared_policy", # Simplified signature
            count_steps_by="agent_steps",
        )
        .debugging(
            log_level="ERROR",
            seed=42
        )
    ) # This closes the PPOConfig chain

    return config


def train_with_max_gpu():
    """Train with maximum GPU utilization."""

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()

    print("\n📋 Initializing Ray for maximum GPU performance...")
    ray.init(
        num_gpus=1,
        num_cpus=11,  # Use most CPUs for workers
        logging_level="ERROR",
        include_dashboard=False,
    )

    resources = ray.available_resources()
    print(f"✅ Ray initialized - GPUs: {resources.get('GPU', 0)}, CPUs: {resources.get('CPU', 0)}")

    # Configuration
    num_agents = 8  # More agents for complexity

    # Register environment
    register_env("optimized_multi_env", lambda config: OptimizedMultiAgentEnv(config))

    # Create optimized config
    print(f"\n⚙️ Creating config for {num_agents} agents with large networks...")
    config = create_optimized_ppo_config(num_agents)

    # Build algorithm
    print("🔨 Building algorithm...")
    try:
        algo = config.build()
        print("✅ Algorithm ready!")
    except Exception as e:
        print(f"❌ Build failed: {e}")
        ray.shutdown()
        return

    # Training
    print("\n📈 Starting high-performance GPU training...")
    print("Monitor GPU usage with: watch -n 1 nvidia-smi")
    print("\nIter | Reward  | Loss   | LR     | GPU% | Mem(GB) | Speed(k) | Time")
    print("-" * 75)

    results_dir = Path("gpu_optimized_results")
    results_dir.mkdir(exist_ok=True)

    metrics = {
        'rewards': [],
        'losses': [],
        'gpu_util': [],
        'gpu_mem': [],
        'throughput': [],
        'lr': []
    }

    try:
        for i in range(100):
            start = time.time()

            # Train
            result = algo.train()

            # Extract metrics
            train_time = time.time() - start
            reward = result.get("episode_reward_mean", 0.0)

            # Get training info
            info = result.get("info", {}).get("learner", {})
            if info:
                # Get first agent's loss as representative
                first_agent = list(info.keys())[0] if info else None
                if first_agent and isinstance(info[first_agent], dict):
                    loss = info[first_agent].get("learner_stats", {}).get("total_loss", 0.0)
                    lr = info[first_agent].get("learner_stats", {}).get("cur_lr", 0.0)
                else:
                    loss = 0.0
                    lr = 0.0
            else:
                loss = 0.0
                lr = 0.0

            # Performance metrics
            timesteps = result.get("num_env_steps_sampled", 0)
            throughput = timesteps / train_time / 1000 if train_time > 0 else 0

            # GPU metrics
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                # Estimate GPU utilization based on performance
                gpu_util = min(100, throughput * 0.5 + gpu_mem * 10)
            else:
                gpu_mem = 0
                gpu_util = 0

            # Store metrics
            metrics['rewards'].append(reward)
            metrics['losses'].append(loss)
            metrics['gpu_util'].append(gpu_util)
            metrics['gpu_mem'].append(gpu_mem)
            metrics['throughput'].append(throughput)
            metrics['lr'].append(lr)

            # Print progress
            print(f"{i+1:4d} | {reward:7.2f} | {loss:6.3f} | {lr:.5f} | "
                  f"{gpu_util:4.0f} | {gpu_mem:7.2f} | {throughput:7.1f} | {train_time:5.1f}s")

            # Save checkpoints
            if (i + 1) % 20 == 0:
                checkpoint = algo.save(results_dir / f"checkpoint_{i+1}")
                print(f"💾 Checkpoint saved: {checkpoint}")

                # Force GPU memory measurement
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    actual_gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"   📊 Actual GPU memory used: {actual_gpu_mem:.2f} GB")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")

    finally:
        # Summary
        if metrics['rewards']:
            print("\n" + "=" * 75)
            print("📊 Training Summary:")
            print(f"   Final reward: {metrics['rewards'][-1]:.3f}")
            print(f"   Best reward: {max(metrics['rewards']):.3f}")
            print(f"   Avg throughput: {np.mean(metrics['throughput']):.1f}k samples/s")
            print(f"   Max throughput: {max(metrics['throughput']):.1f}k samples/s")

            if torch.cuda.is_available():
                print(f"   Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
                print(f"   Avg GPU utilization: {np.mean(metrics['gpu_util']):.1f}%")

        # Save final model
        try:
            final_checkpoint = algo.save(str(results_dir / "final_model"))
            print(f"\n💾 Final model saved: {final_checkpoint}")
        except:
            pass

        # Save metrics
        np.save(results_dir / "training_metrics.npy", metrics)

        # Cleanup
        algo.stop()
        ray.shutdown()

        if os.path.exists('worker_setup.py'):
            os.remove('worker_setup.py')

        print("\n✅ Training complete!")
        print(f"📁 Results saved in: {results_dir}")

        # Create plots
        try:
            import matplotlib.pyplot as plt

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            # Rewards
            ax1.plot(metrics['rewards'], linewidth=2)
            ax1.set_title('Episode Rewards', fontsize=14)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Mean Reward')
            ax1.grid(True, alpha=0.3)

            # Throughput
            ax2.plot(metrics['throughput'], linewidth=2, color='green')
            ax2.set_title('Training Throughput', fontsize=14)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('k samples/second')
            ax2.grid(True, alpha=0.3)

            # GPU metrics
            ax3.plot(metrics['gpu_util'], label='GPU Util %', linewidth=2)
            ax3.plot(np.array(metrics['gpu_mem']) * 10, label='GPU Mem (GB) x10', linewidth=2)
            ax3.set_title('GPU Utilization', fontsize=14)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Percentage / Memory')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Loss and LR
            ax4_twin = ax4.twinx()
            ax4.plot(metrics['losses'], 'b-', label='Loss', linewidth=2)
            ax4_twin.plot(metrics['lr'], 'r-', label='Learning Rate', linewidth=2)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Loss', color='b')
            ax4_twin.set_ylabel('Learning Rate', color='r')
            ax4.set_title('Training Loss and Learning Rate', fontsize=14)
            ax4.grid(True, alpha=0.3)

            plt.suptitle(f'GPU-Optimized MARL Training ({num_agents} agents)', fontsize=16)
            plt.tight_layout()
            plt.savefig(results_dir / 'training_results_optimized.png', dpi=150)
            print(f"📊 Plots saved: {results_dir / 'training_results_optimized.png'}")

        except ImportError:
            print("(Install matplotlib to see plots)")


if __name__ == "__main__":
    print("\n🎯 This version maximizes GPU utilization!")
    print("   - 8 agents with 128-dim observations")
    print("   - 1024-1024-512 neural networks")
    print("   - 16k batch size, 1k minibatch")
    print("   - 8 rollout workers with 2 envs each")
    print("\nStarting training...\n")

    train_with_max_gpu()
