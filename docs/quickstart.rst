# File: docs/tutorials/quickstart.rst
"""
Quick start tutorial for new users.
"""

Quick Start Tutorial
===================

This tutorial gets you up and running with Nature MARL in 15 minutes.

🎯 Learning Objectives
---------------------

By the end of this tutorial, you'll be able to:

- Create a bio-inspired multi-agent RL system
- Train agents with pheromone communication
- Monitor bio-inspired metrics
- Visualize emergent behaviors

📋 Prerequisites
----------------

- Basic Python knowledge
- Familiarity with reinforcement learning concepts
- Nature MARL installed (see :doc:`../installation`)

🚀 Your First Bio-Inspired Agents
---------------------------------

Let's create agents that communicate using pheromone-like signals:

.. code-block:: python

   import torch
   from gymnasium.spaces import Box, Discrete
   from nature_marl import create_production_bio_module_spec
   from ray.rllib.algorithms.ppo import PPOConfig

   # Step 1: Define observation and action spaces
   observation_space = Box(low=-1.0, high=1.0, shape=(8,))
   action_space = Discrete(4)

   # Step 2: Create bio-inspired module specification
   module_spec = create_production_bio_module_spec(
       obs_space=observation_space,
       act_space=action_space,
       num_agents=6,
       use_communication=True,
       model_config={
           "hidden_dim": 256,
           "memory_dim": 64,
           "comm_channels": 16,
           "comm_rounds": 3,
           "use_positional_encoding": True,
           "adaptive_plasticity": True,
           "debug_mode": True  # Enable detailed logging
       }
   )

   print("✅ Bio-inspired module specification created!")

🧠 Understanding the Configuration
----------------------------------

Let's break down the key bio-inspired parameters:

.. code-block:: python

   config_explanation = {
       "num_agents": 6,                    # Number of agents in the swarm
       "use_communication": True,          # Enable pheromone-like communication
       "hidden_dim": 256,                  # Neural network size
       "memory_dim": 64,                   # Memory capacity (neural plasticity)
       "comm_channels": 16,                # Pheromone signal complexity
       "comm_rounds": 3,                   # Multi-round communication
       "use_positional_encoding": True,    # Spatial awareness (like bee navigation)
       "adaptive_plasticity": True,        # Learning rate adapts to signal strength
       "debug_mode": True                  # Detailed bio-inspired metrics
   }

🌍 Setting Up an Environment
----------------------------

Create a simple environment to test your bio-inspired agents:

.. code-block:: python

   from pettingzoo.mpe import simple_spread_v3
   from supersuit import concat_vec_envs_v1, pettingzoo_env_to_vec_env_v1

   # Create multi-agent environment
   def create_environment():
       env = simple_spread_v3.parallel_env(
           N=6,           # 6 agents (matches our module config)
           local_ratio=0.5,
           max_cycles=25,
           continuous_actions=False
       )

       # Convert for RLlib
       env = pettingzoo_env_to_vec_env_v1(env)
       env = concat_vec_envs_v1(env, 1, num_cpus=1, base_class="gym")

       return env

   print("✅ Environment created!")

🤖 Training Bio-Inspired Agents
-------------------------------

Now let's put it all together and train your agents:

.. code-block:: python

   import ray
   from ray import tune

   # Initialize Ray
   ray.init(local_mode=True)  # Use local mode for debugging

   # Create PPO configuration with bio-inspired agents
   config = (
       PPOConfig()
       .environment(
           env=create_environment,
           env_config={}
       )
       .framework("torch")
       .rl_module(rl_module_spec=module_spec)
       .training(
           train_batch_size_per_learner=512,
           minibatch_size=128,
           lr=3e-4,
           num_epochs=10,
           gamma=0.99
       )
       .env_runners(
           num_env_runners=2,
           num_envs_per_env_runner=1
       )
       # Shared policy - all agents use the same bio-inspired network
       .multi_agent(
           policies={"shared_policy": None},
           policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
       )
   )

   # Build and train
   algorithm = config.build()

   print("🚀 Starting bio-inspired training...")

   for iteration in range(10):
       result = algorithm.train()

       print(f"Iteration {iteration + 1}:")
       print(f"  Reward: {result['env_runners']['episode_reward_mean']:.2f}")
       print(f"  Length: {result['env_runners']['episode_len_mean']:.1f}")

       # Access bio-inspired metrics (if available)
       if 'comm_entropy' in result.get('learner', {}):
           print(f"  Communication Entropy: {result['learner']['comm_entropy']:.3f}")
           print(f"  Attention Entropy: {result['learner']['attention_entropy']:.3f}")

   print("✅ Training completed!")

📊 Monitoring Bio-Inspired Behavior
-----------------------------------

Let's analyze what our agents learned:

.. code-block:: python

   from nature_marl.utils.logging_config import BioInspiredMetricsTracker

   # Create metrics tracker
   metrics_tracker = BioInspiredMetricsTracker(
       track_attention=True,
       track_communication=True,
       track_plasticity=True
   )

   # Get a sample from the trained policy
   env = create_environment()
   obs, info = env.reset()

   # Run one step to get bio-inspired outputs
   action_dict = algorithm.compute_actions(obs)

   # Access the RL module to get bio-inspired metrics
   rl_module = algorithm.get_policy().model

   # Convert observations for the module
   obs_tensor = torch.FloatTensor(obs)
   batch = {"obs": obs_tensor}

   # Forward pass to get bio-inspired outputs
   with torch.no_grad():
       output = rl_module.forward_inference(batch)

   # Analyze bio-inspired behavior
   if "comm_signal" in output:
       comm_signal = output["comm_signal"]
       print(f"📡 Communication signal shape: {comm_signal.shape}")
       print(f"📊 Signal strength: {torch.norm(comm_signal).item():.3f}")

   if "attention_weights" in output:
       attention_weights = output["attention_weights"][0]  # First comm round
       print(f"👁️  Attention pattern shape: {attention_weights.shape}")

       # Who's talking to whom?
       avg_attention = attention_weights.mean(dim=(0, 1))  # Average over batch and heads
       print("🗣️  Agent communication patterns:")
       for i in range(min(6, avg_attention.shape[0])):
           for j in range(min(6, avg_attention.shape[1])):
               if avg_attention[i, j] > 0.2:  # Strong communication
                   print(f"   Agent {i} → Agent {j}: {avg_attention[i, j]:.3f}")

🎨 Visualizing Emergent Behavior
--------------------------------

Create simple visualizations of your agents' behavior:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   def visualize_attention_patterns(attention_weights):
       """Visualize who communicates with whom."""
       # Average over batch and heads
       avg_attention = attention_weights[0].mean(dim=(0, 1)).cpu().numpy()

       plt.figure(figsize=(8, 6))
       plt.imshow(avg_attention, cmap='viridis', interpolation='nearest')
       plt.colorbar(label='Attention Weight')
       plt.title('Agent Communication Patterns\n(Pheromone-Inspired Attention)')
       plt.xlabel('Target Agent')
       plt.ylabel('Source Agent')

       # Add text annotations
       for i in range(avg_attention.shape[0]):
           for j in range(avg_attention.shape[1]):
               plt.text(j, i, f'{avg_attention[i, j]:.2f}',
                       ha='center', va='center',
                       color='white' if avg_attention[i, j] > 0.5 else 'black')

       plt.tight_layout()
       plt.show()

   def visualize_communication_signals(comm_signals):
       """Visualize pheromone-like signals."""
       signals = comm_signals.cpu().numpy()

       plt.figure(figsize=(12, 4))

       # Plot signal strength over agents
       plt.subplot(1, 2, 1)
       signal_strength = np.linalg.norm(signals, axis=1)
       plt.bar(range(len(signal_strength)), signal_strength)
       plt.title('Pheromone Signal Strength by Agent')
       plt.xlabel('Agent ID')
       plt.ylabel('Signal Strength')

       # Plot signal composition
       plt.subplot(1, 2, 2)
       plt.imshow(signals.T, cmap='RdBu', aspect='auto')
       plt.colorbar(label='Signal Value')
       plt.title('Pheromone Signal Composition')
       plt.xlabel('Agent ID')
       plt.ylabel('Communication Channel')

       plt.tight_layout()
       plt.show()

   # Visualize your agents' behavior
   if "attention_weights" in output:
       visualize_attention_patterns(output["attention_weights"])

   if "comm_signal" in output:
       visualize_communication_signals(output["comm_signal"])

🎯 What You've Accomplished
--------------------------

Congratulations! You've successfully:

✅ **Created bio-inspired agents** with pheromone communication
✅ **Trained a multi-agent system** using neural plasticity
✅ **Monitored emergent behaviors** through bio-inspired metrics
✅ **Visualized communication patterns** and signal flow

🚀 Next Steps
-------------

Ready to explore more? Try these advanced topics:

1. **Custom Environments**: :doc:`../advanced/custom_environments`
2. **Hyperparameter Tuning**: :doc:`../advanced/hyperparameter_tuning`
3. **Research Applications**: :doc:`../advanced/research_applications`
4. **Performance Optimization**: :doc:`../performance_optimization`

💡 **Experiment Ideas**:

- Try different numbers of agents (2-16)
- Adjust communication rounds and channels
- Enable/disable positional encoding
- Compare with traditional MARL methods
- Create custom reward functions that encourage cooperation

**Pro Tip**: Enable `debug_mode=True` in your model config to get detailed
bio-inspired metrics that help you understand what your agents are learning!

🤝 Need Help?
------------

- **Documentation**: Continue reading our guides
- **Examples**: Check out :doc:`../examples/index`
- **Community**: Join our GitHub discussions
- **Issues**: Report bugs or request features

Happy training with bio-inspired intelligence! 🌿🤖
