# File: tests/test_enhanced_communication.py
#!/usr/bin/env python3

"""
Comprehensive test suite for the enhanced nature-inspired communication module.
Tests functionality, compatibility, and performance characteristics.
"""

import unittest
import torch
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

# Import the enhanced module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from marlcomm.rl_module import (
    NatureInspiredCommModule,
    MultiRoundAttentionComm,
    PheromoneEncoder,
    NeuralPlasticityMemory,
    get_nature_comm_config,
    create_enhanced_nature_comm_module_spec
)


class TestNatureInspiredCommModule(unittest.TestCase):
    """Test suite for the main communication module."""

    def setUp(self):
        """Set up test environment and module."""
        self.obs_space = Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.action_space = Discrete(5)
        self.num_agents = 4

        self.model_config = get_nature_comm_config(
            num_agents=self.num_agents,
            embed_dim=64,
            comm_rounds=2,
            attention_heads=4
        )

        self.module = NatureInspiredCommModule(
            observation_space=self.obs_space,
            action_space=self.action_space,
            model_config=self.model_config
        )
        self.module.setup()

    def test_module_initialization(self):
        """Test that the module initializes correctly."""
        # Check that all components are created
        self.assertIsNotNone(self.module.obs_encoder)
        self.assertIsNotNone(self.module.pheromone_encoder)
        self.assertIsNotNone(self.module.attention_comm)
        self.assertIsNotNone(self.module.memory_system)
        self.assertIsNotNone(self.module.policy_net)
        self.assertIsNotNone(self.module.value_net)

        # Check dimensions
        self.assertEqual(self.module.num_agents, self.num_agents)
        self.assertEqual(self.module.embed_dim, 64)
        self.assertEqual(self.module.pheromone_dim, 4)  # embed_dim // 16

    def test_forward_pass_single_agent(self):
        """Test forward pass with single agent."""
        single_config = get_nature_comm_config(num_agents=1, embed_dim=64)
        single_module = NatureInspiredCommModule(
            observation_space=self.obs_space,
            action_space=self.action_space,
            model_config=single_config
        )
        single_module.setup()

        batch_size = 8
        obs = torch.randn(batch_size, self.obs_space.shape[0])
        batch = {"obs": obs}

        with torch.no_grad():
            output = single_module._forward(batch)

        # Check output structure
        self.assertIn("action_dist_inputs", output)
        self.assertIn("vf_preds", output)
        self.assertIn("pheromone_signals", output)
        self.assertIn("communication_stats", output)

        # Check output shapes
        self.assertEqual(output["action_dist_inputs"].shape, (batch_size, self.action_space.n))
        self.assertEqual(output["vf_preds"].shape, (batch_size,))
        self.assertEqual(output["pheromone_signals"].shape, (batch_size, single_config["pheromone_dim"]))

    def test_forward_pass_multi_agent(self):
        """Test forward pass with multiple agents."""
        batch_size = self.num_agents * 4  # 4 environments, each with num_agents
        true_obs_dim = self.obs_space.shape[0] * self.num_agents
        obs = torch.randn(batch_size, true_obs_dim)
        batch = {"obs": obs}

        with torch.no_grad():
            output = self.module._forward(batch)

        # Check output structure and shapes
        self.assertIn("action_dist_inputs", output)
        self.assertIn("vf_preds", output)
        self.assertIn("pheromone_signals", output)

        self.assertEqual(output["action_dist_inputs"].shape, (batch_size, self.action_space.n))
        self.assertEqual(output["vf_preds"].shape, (batch_size,))

    def test_communication_loss(self):
        """Test communication loss computation."""
        batch_size = 8
        obs = torch.randn(batch_size, self.obs_space.shape[0] * self.num_agents)
        batch = {"obs": obs}

        # Forward pass to generate communication stats
        with torch.no_grad():
            self.module._forward(batch)

        # Test loss computation
        comm_loss = self.module.get_communication_loss(
            entropy_coeff=0.01,
            sparsity_coeff=0.001,
            diversity_coeff=0.01
        )

        self.assertIsInstance(comm_loss, torch.Tensor)
        self.assertEqual(comm_loss.dim(), 0)  # Scalar loss

    def test_memory_persistence(self):
        """Test that memory state persists between forward passes."""
        batch_size = 4
        obs = torch.randn(batch_size, self.obs_space.shape[0] * self.num_agents)
        batch = {"obs": obs}

        # First forward pass
        with torch.no_grad():
            output1 = self.module._forward(batch)
            initial_memory = self.module.memory_state.clone()

        # Second forward pass
        with torch.no_grad():
            output2 = self.module._forward(batch)
            updated_memory = self.module.memory_state.clone()

        # Memory should have changed
        self.assertFalse(torch.allclose(initial_memory, updated_memory))

    def test_gradient_flow(self):
        """Test that gradients flow through the module correctly."""
        batch_size = 4
        obs = torch.randn(batch_size, self.obs_space.shape[0] * self.num_agents)
        batch = {"obs": obs}

        # Enable gradients
        obs.requires_grad_(True)

        output = self.module._forward(batch)
        loss = output["action_dist_inputs"].mean() + output["vf_preds"].mean()
        loss.backward()

        # Check that gradients exist
        self.assertIsNotNone(obs.grad)
        self.assertTrue(torch.any(obs.grad != 0))


class TestCommunicationComponents(unittest.TestCase):
    """Test individual communication components."""

    def test_multi_round_attention(self):
        """Test multi-round attention communication."""
        embed_dim = 64
        num_heads = 8
        num_rounds = 3
        num_agents = 4
        batch_size = 8

        attention_comm = MultiRoundAttentionComm(embed_dim, num_heads, num_rounds)

        # Test input
        embeddings = torch.randn(batch_size * num_agents, embed_dim)

        with torch.no_grad():
            final_embeddings, attention_weights = attention_comm(embeddings, num_agents)

        # Check output shapes
        self.assertEqual(final_embeddings.shape, (batch_size * num_agents, embed_dim))
        self.assertEqual(attention_weights.shape, (batch_size, num_heads, num_agents, num_agents))

    def test_pheromone_encoder(self):
        """Test pheromone signal encoding."""
        input_dim = 128
        pheromone_dim = 16
        batch_size = 8

        encoder = PheromoneEncoder(input_dim, pheromone_dim)

        # Test encoding
        x = torch.randn(batch_size, input_dim)
        with torch.no_grad():
            pheromones = encoder(x)

        # Check output properties
        self.assertEqual(pheromones.shape, (batch_size, pheromone_dim))
        self.assertTrue(torch.all(pheromones >= -1))  # Tanh bounded
        self.assertTrue(torch.all(pheromones <= 1))

    def test_neural_plasticity_memory(self):
        """Test neural plasticity memory system."""
        input_dim = 80
        memory_dim = 32
        batch_size = 4

        memory_system = NeuralPlasticityMemory(input_dim, memory_dim)

        # Test memory update
        experience = torch.randn(batch_size, input_dim)
        hidden_state = torch.zeros(batch_size, 2, memory_dim)  # 2 layers

        with torch.no_grad():
            memory_output, new_hidden = memory_system(experience, hidden_state)

        # Check output shapes
        self.assertEqual(memory_output.shape, (batch_size, memory_dim))
        self.assertEqual(new_hidden.shape, (batch_size, 2, memory_dim))

        # Memory should change
        self.assertFalse(torch.allclose(hidden_state, new_hidden))


class TestConfigurationHelpers(unittest.TestCase):
    """Test configuration helper functions."""

    def test_get_nature_comm_config(self):
        """Test configuration generation."""
        config = get_nature_comm_config(
            num_agents=8,
            embed_dim=256,
            comm_rounds=3,
            attention_heads=8
        )

        expected_keys = [
            "num_agents", "embed_dim", "pheromone_dim", "memory_dim",
            "num_comm_rounds", "num_attention_heads", "fcnet_hiddens", "fcnet_activation"
        ]

        for key in expected_keys:
            self.assertIn(key, config)

        # Check derived values
        self.assertEqual(config["pheromone_dim"], 256 // 16)
        self.assertEqual(config["memory_dim"], 256 // 4)

    def test_module_spec_creation(self):
        """Test RLModuleSpec creation."""
        obs_space = Box(low=-1, high=1, shape=(10,))
        action_space = Discrete(5)
        model_config = get_nature_comm_config()

        spec = create_enhanced_nature_comm_module_spec(
            obs_space, action_space, model_config
        )

        # Check spec properties
        self.assertEqual(spec.module_class, NatureInspiredCommModule)
        self.assertEqual(spec.observation_space, obs_space)
        self.assertEqual(spec.action_space, action_space)
        self.assertEqual(spec.model_config_dict, model_config)


class TestCompatibility(unittest.TestCase):
    """Test compatibility with RLlib components."""

    def test_rllib_compatibility(self):
        """Test basic RLlib compatibility."""
        try:
            from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
            from ray.rllib.algorithms.ppo import PPOConfig

            # Test spec creation
            obs_space = Box(low=-1, high=1, shape=(10,))
            action_space = Discrete(5)

            spec = SingleAgentRLModuleSpec(
                module_class=NatureInspiredCommModule,
                observation_space=obs_space,
                action_space=action_space,
                model_config_dict=get_nature_comm_config()
            )

            # Test config integration
            config = PPOConfig().rl_module(rl_module_spec=spec)

            self.assertIsNotNone(config)

        except ImportError as e:
            self.skipTest(f"RLlib not available: {e}")


def run_performance_benchmark():
    """Run performance benchmarks for the communication module."""
    print("\n=== Performance Benchmark ===")

    # Setup
    obs_space = Box(low=-1, high=1, shape=(20,), dtype=np.float32)
    action_space = Discrete(10)
    num_agents = 8

    model_config = get_nature_comm_config(
        num_agents=num_agents,
        embed_dim=256,
        comm_rounds=3,
        attention_heads=8
    )

    module = NatureInspiredCommModule(
        observation_space=obs_space,
        action_space=action_space,
        model_config=model_config
    )
    module.setup()

    # Benchmark different batch sizes
    batch_sizes = [16, 32, 64, 128]

    for batch_size in batch_sizes:
        obs_dim = obs_space.shape[0] * num_agents
        obs = torch.randn(batch_size, obs_dim)
        batch = {"obs": obs}

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                module._forward(batch)

        # Timing
        import time
        start_time = time.time()

        for _ in range(100):
            with torch.no_grad():
                output = module._forward(batch)

        end_time = time.time()
        avg_time = (end_time - start_time) / 100

        print(f"Batch size {batch_size}: {avg_time*1000:.2f}ms per forward pass")

    print("=== Benchmark Complete ===\n")


if __name__ == "__main__":
    # Run standard tests
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run performance benchmark
    run_performance_benchmark()
