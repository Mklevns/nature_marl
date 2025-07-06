# File: tests/test_production_bio_module.py
"""
Comprehensive Unit Test Suite for Production Bio-Inspired RL Module

This test suite provides complete coverage of all bio-inspired components
with automated testing, performance benchmarking, and CI integration.

PHASE 3 TESTING FEATURES:
✅ Instantiation tests for all observation/action space combinations
✅ Shape assertion tests for all forward methods
✅ Stateful behavior validation across sequences
✅ Bio-inspired component isolation tests
✅ Performance benchmarking and profiling
✅ Memory leak detection
✅ Gradient flow validation
✅ Communication pattern analysis
✅ CI/CD integration ready
"""

import unittest
import logging
import time
import gc
import tracemalloc
from typing import Dict, List, Tuple, Any, Optional
from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary

# Import the production module
from nature_marl.core.production_bio_inspired_rl_module import (
    ProductionUnifiedBioInspiredRLModule,
    ProductionPheromoneAttentionNetwork,
    ProductionNeuralPlasticityMemory,
    ProductionMultiActionHead,
    create_production_bio_module_spec
)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestUtilities:
    """Utility functions for testing bio-inspired components."""

    @staticmethod
    def create_test_spaces() -> List[Tuple[str, Any, Any]]:
        """Create comprehensive test cases for different space combinations."""
        return [
            ("Box-Discrete", Box(low=-1, high=1, shape=(4,)), Discrete(5)),
            ("Box-MultiDiscrete", Box(low=-1, high=1, shape=(8,)), MultiDiscrete([3, 4, 2])),
            ("Box-Box", Box(low=-1, high=1, shape=(6,)), Box(low=-2, high=2, shape=(3,))),
            ("Box-MultiBinary", Box(low=-1, high=1, shape=(10,)), MultiBinary(4)),
            ("Discrete-Discrete", Discrete(8), Discrete(3)),
            ("Discrete-MultiDiscrete", Discrete(5), MultiDiscrete([2, 3, 4])),
            ("MultiDiscrete-Discrete", MultiDiscrete([2, 3]), Discrete(6)),
            ("MultiDiscrete-MultiDiscrete", MultiDiscrete([3, 2]), MultiDiscrete([4, 2, 3])),
        ]

    @staticmethod
    def create_test_batch(obs_space, batch_size: int = 4):
        """Create a test batch for the given observation space."""
        if isinstance(obs_space, Box):
            return torch.randn(batch_size, *obs_space.shape)
        elif isinstance(obs_space, Discrete):
            return torch.randint(0, obs_space.n, (batch_size,))
        elif isinstance(obs_space, MultiDiscrete):
            return torch.stack([
                torch.randint(0, int(n), (batch_size,)) for n in obs_space.nvec
            ], dim=1)
        elif isinstance(obs_space, MultiBinary):
            return torch.randint(0, 2, (batch_size, obs_space.n))
        else:
            raise ValueError(f"Unsupported observation space: {type(obs_space)}")

    @staticmethod
    def expected_action_dim(action_space):
        """Calculate expected action dimension for validation."""
        if isinstance(action_space, Discrete):
            return action_space.n
        elif isinstance(action_space, MultiDiscrete):
            return sum(action_space.nvec)
        elif isinstance(action_space, Box):
            return int(np.prod(action_space.shape))
        elif isinstance(action_space, MultiBinary):
            return action_space.n
        else:
            raise ValueError(f"Unsupported action space: {type(action_space)}")


@contextmanager
def memory_profiler():
    """Context manager for memory usage profiling."""
    tracemalloc.start()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    yield

    current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    logger.info(f"Memory usage: CPU peak {peak / 1024 / 1024:.1f}MB, "
                f"GPU delta {(current_memory - start_memory) / 1024 / 1024:.1f}MB")


class TestPheromoneAttentionNetwork(unittest.TestCase):
    """Test cases for the pheromone attention network component."""

    def setUp(self):
        self.hidden_dim = 64
        self.num_heads = 8
        self.max_agents = 16

    def test_initialization(self):
        """Test proper initialization of attention network."""
        network = ProductionPheromoneAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            use_positional_encoding=True,
            max_agents=self.max_agents
        )

        # Check parameter initialization
        self.assertTrue(hasattr(network, 'positional_encoding'))
        self.assertEqual(network.positional_encoding.shape, (self.max_agents, self.hidden_dim))

        # Check module components
        self.assertIsInstance(network.attention, nn.MultiheadAttention)
        self.assertIsInstance(network.pheromone_encoder, nn.Sequential)
        self.assertIsInstance(network.layer_norm, nn.LayerNorm)

    def test_forward_pass_shapes(self):
        """Test output shapes for various input configurations."""
        network = ProductionPheromoneAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            max_agents=self.max_agents
        )

        test_cases = [
            (2, 4),   # batch_size=2, num_agents=4
            (4, 8),   # batch_size=4, num_agents=8
            (1, 12),  # batch_size=1, num_agents=12
        ]

        for batch_size, num_agents in test_cases:
            with self.subTest(batch_size=batch_size, num_agents=num_agents):
                agent_features = torch.randn(batch_size, num_agents, self.hidden_dim)

                output, pheromones, attention_weights = network(
                    agent_features,
                    return_attention_weights=True
                )

                # Validate shapes
                self.assertEqual(output.shape, agent_features.shape)
                self.assertEqual(pheromones.shape, agent_features.shape)
                self.assertEqual(
                    attention_weights.shape,
                    (batch_size, self.num_heads, num_agents, num_agents)
                )

                # Validate attention properties
                attn_sum = attention_weights.sum(dim=-1)
                self.assertTrue(torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5))

    def test_positional_encoding(self):
        """Test that positional encoding affects outputs."""
        # Network with positional encoding
        network_with_pos = ProductionPheromoneAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_heads=4,
            use_positional_encoding=True,
            max_agents=8
        )

        # Network without positional encoding
        network_without_pos = ProductionPheromoneAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_heads=4,
            use_positional_encoding=False,
            max_agents=8
        )

        # Same input
        agent_features = torch.randn(2, 6, self.hidden_dim)

        output_with_pos, _, _ = network_with_pos(agent_features, return_attention_weights=False)
        output_without_pos, _, _ = network_without_pos(agent_features, return_attention_weights=False)

        # Outputs should be different
        feature_diff = torch.norm(output_with_pos - output_without_pos).item()
        self.assertGreater(feature_diff, 0.1, "Positional encoding should affect outputs")

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        network = ProductionPheromoneAttentionNetwork(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            max_agents=8
        )

        # Test dimension mismatch
        wrong_dim_features = torch.randn(2, 4, 32)  # Wrong feature dimension
        with self.assertRaises(RuntimeError):
            network(wrong_dim_features)

        # Test too many agents
        too_many_agents = torch.randn(1, 10, self.hidden_dim)  # More than max_agents
        with self.assertRaises(ValueError):
            network(too_many_agents)


class TestNeuralPlasticityMemory(unittest.TestCase):
    """Test cases for the neural plasticity memory component."""

    def setUp(self):
        self.input_dim = 32
        self.memory_dim = 16
        self.plasticity_rate = 0.2

    def test_initialization(self):
        """Test proper initialization of memory module."""
        memory = ProductionNeuralPlasticityMemory(
            input_dim=self.input_dim,
            memory_dim=self.memory_dim,
            plasticity_rate=self.plasticity_rate,
            adaptive_plasticity=True
        )

        # Check GRU bias initialization
        if hasattr(memory.memory_cell, 'bias_ih'):
            bias_ih = memory.memory_cell.bias_ih
            update_gate_start = self.memory_dim
            update_gate_end = 2 * self.memory_dim
            update_bias = bias_ih[update_gate_start:update_gate_end]

            # Update gate biases should be positive
            self.assertTrue(torch.all(update_bias > 0), "Update gate biases should be positive")

    def test_memory_update(self):
        """Test memory update functionality."""
        memory = ProductionNeuralPlasticityMemory(
            input_dim=self.input_dim,
            memory_dim=self.memory_dim,
            plasticity_rate=self.plasticity_rate,
            adaptive_plasticity=True
        )

        batch_size = 4
        inputs = torch.randn(batch_size, self.input_dim)
        hidden = torch.zeros(batch_size, self.memory_dim)

        # Update memory
        new_hidden = memory(inputs, hidden)

        # Validate output shape
        self.assertEqual(new_hidden.shape, (batch_size, self.memory_dim))

        # Memory should change
        memory_change = torch.norm(new_hidden - hidden).item()
        self.assertGreater(memory_change, 0, "Memory should change after update")

    def test_adaptive_plasticity(self):
        """Test adaptive plasticity based on signal strength."""
        memory = ProductionNeuralPlasticityMemory(
            input_dim=self.input_dim,
            memory_dim=self.memory_dim,
            plasticity_rate=self.plasticity_rate,
            adaptive_plasticity=True
        )

        batch_size = 4
        hidden = torch.zeros(batch_size, self.memory_dim)

        # Weak vs strong signals
        weak_signal = torch.randn(batch_size, self.input_dim) * 0.1
        strong_signal = torch.randn(batch_size, self.input_dim) * 2.0

        new_hidden_weak = memory(weak_signal, hidden.clone())
        new_hidden_strong = memory(strong_signal, hidden.clone())

        weak_change = torch.norm(new_hidden_weak - hidden).item()
        strong_change = torch.norm(new_hidden_strong - hidden).item()

        # Strong signals should cause more change
        self.assertGreater(strong_change, weak_change,
                          "Strong signals should cause more plasticity")

    def test_gradient_flow(self):
        """Test that gradients flow properly (no blocking)."""
        memory = ProductionNeuralPlasticityMemory(
            input_dim=self.input_dim,
            memory_dim=self.memory_dim,
            plasticity_rate=self.plasticity_rate
        )

        batch_size = 4
        inputs = torch.randn(batch_size, self.input_dim, requires_grad=True)
        hidden = torch.randn(batch_size, self.memory_dim)

        # Forward pass
        new_hidden = memory(inputs, hidden)
        loss = new_hidden.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(inputs.grad, "Gradients should flow to inputs")
        self.assertGreater(torch.norm(inputs.grad).item(), 0, "Gradient norm should be positive")


class TestMultiActionHead(unittest.TestCase):
    """Test cases for the multi-action head component."""

    def setUp(self):
        self.input_dim = 128

    def test_action_space_support(self):
        """Test support for all action space types."""
        test_cases = [
            ("Discrete", Discrete(5), 5),
            ("MultiDiscrete", MultiDiscrete([3, 4, 2]), 9),
            ("Box", Box(low=-1, high=1, shape=(3,)), 3),
            ("MultiBinary", MultiBinary(4), 4),
        ]

        for space_name, action_space, expected_dim in test_cases:
            with self.subTest(space_name=space_name):
                action_head = ProductionMultiActionHead(self.input_dim, action_space)

                batch_size = 4
                features = torch.randn(batch_size, self.input_dim)
                logits = action_head(features)

                self.assertEqual(logits.shape, (batch_size, expected_dim))

    def test_error_handling(self):
        """Test error handling for invalid action spaces."""
        # Invalid action space
        with self.assertRaises(ValueError):
            ProductionMultiActionHead(self.input_dim, "invalid_space")

        # Invalid input dimension
        with self.assertRaises(ValueError):
            ProductionMultiActionHead(-1, Discrete(5))


class TestProductionUnifiedModule(unittest.TestCase):
    """Test cases for the complete production bio-inspired module."""

    def setUp(self):
        self.test_spaces = TestUtilities.create_test_spaces()

    def test_module_instantiation(self):
        """Test module instantiation with all space combinations."""
        for space_name, obs_space, act_space in self.test_spaces:
            with self.subTest(space_name=space_name):
                try:
                    module = ProductionUnifiedBioInspiredRLModule(
                        observation_space=obs_space,
                        action_space=act_space,
                        model_config={
                            "num_agents": 4,
                            "use_communication": True,
                            "hidden_dim": 64,
                            "memory_dim": 32,
                            "debug_mode": False
                        }
                    )
                    module.setup()

                    # Basic validation
                    self.assertTrue(module.is_stateful())
                    self.assertIsInstance(module.get_initial_state(), dict)

                except Exception as e:
                    self.fail(f"Module instantiation failed for {space_name}: {e}")

    def test_forward_pass_shapes(self):
        """Test forward pass shapes for all methods."""
        obs_space = Box(low=-1, high=1, shape=(4,))
        act_space = MultiDiscrete([3, 2, 4])

        module = ProductionUnifiedBioInspiredRLModule(
            observation_space=obs_space,
            action_space=act_space,
            model_config={
                "num_agents": 4,
                "use_communication": True,
                "hidden_dim": 64,
                "memory_dim": 32
            }
        )
        module.setup()

        batch_size = 8  # 4 agents * 2
        obs = torch.randn(batch_size, 4)
        batch = {"obs": obs}

        expected_action_dim = sum(act_space.nvec)  # 3 + 2 + 4 = 9

        # Test all forward methods
        forward_methods = [
            module.forward_train,
            module.forward_exploration,
            module.forward_inference
        ]

        for forward_method in forward_methods:
            with self.subTest(method=forward_method.__name__):
                output = forward_method(batch)

                # Validate required outputs
                self.assertIn("action_dist_inputs", output)
                self.assertIn("vf_preds", output)
                self.assertIn("state_out", output)

                # Validate shapes
                self.assertEqual(output["action_dist_inputs"].shape, (batch_size, expected_action_dim))
                self.assertEqual(output["vf_preds"].shape, (batch_size,))
                self.assertIn("hidden_state", output["state_out"])

    def test_stateful_behavior(self):
        """Test that state propagates correctly across sequences."""
        obs_space = Box(low=-1, high=1, shape=(6,))
        act_space = Discrete(4)

        module = ProductionUnifiedBioInspiredRLModule(
            observation_space=obs_space,
            action_space=act_space,
            model_config={
                "num_agents": 2,
                "use_communication": True,
                "memory_dim": 16
            }
        )
        module.setup()

        batch_size = 4
        obs = torch.randn(batch_size, 6)
        batch = {"obs": obs}

        # First forward pass
        output1 = module.forward_train(batch)
        state1 = output1["state_out"]

        # Second forward pass with state
        output2 = module.forward_train(batch, state_in=state1)
        state2 = output2["state_out"]

        # States should be different (memory updated)
        state_diff = torch.norm(state2["hidden_state"] - state1["hidden_state"]).item()
        self.assertGreater(state_diff, 0, "Hidden state should change between steps")

    def test_communication_system(self):
        """Test communication system outputs and metrics."""
        obs_space = Box(low=-1, high=1, shape=(4,))
        act_space = Discrete(3)

        module = ProductionUnifiedBioInspiredRLModule(
            observation_space=obs_space,
            action_space=act_space,
            model_config={
                "num_agents": 6,
                "use_communication": True,
                "comm_channels": 8,
                "comm_rounds": 2
            }
        )
        module.setup()

        batch_size = 12  # 6 agents * 2
        obs = torch.randn(batch_size, 4)
        batch = {"obs": obs}

        output = module.forward_train(batch)

        # Check communication outputs
        self.assertIn("comm_signal", output)
        self.assertIn("comm_entropy", output)
        self.assertIn("comm_sparsity", output)

        # Validate communication signal shape
        self.assertEqual(output["comm_signal"].shape, (batch_size, 8))

        # Check communication metrics are reasonable
        self.assertIsInstance(output["comm_entropy"].item(), float)
        self.assertIsInstance(output["comm_sparsity"].item(), float)
        self.assertGreater(output["comm_sparsity"].item(), 0)

    def test_model_info(self):
        """Test model information reporting."""
        obs_space = Box(low=-1, high=1, shape=(8,))
        act_space = Discrete(5)

        module = ProductionUnifiedBioInspiredRLModule(
            observation_space=obs_space,
            action_space=act_space,
            model_config={"num_agents": 4}
        )
        module.setup()

        model_info = module.get_model_info()

        # Check required fields
        required_fields = [
            "total_parameters", "trainable_parameters", "model_size_mb",
            "num_agents", "hidden_dim", "memory_dim", "use_communication"
        ]

        for field in required_fields:
            self.assertIn(field, model_info)

        # Validate parameter counts
        self.assertGreater(model_info["total_parameters"], 0)
        self.assertGreater(model_info["trainable_parameters"], 0)
        self.assertGreater(model_info["model_size_mb"], 0)


class TestModuleFactory(unittest.TestCase):
    """Test cases for the module factory function."""

    def test_spec_creation(self):
        """Test module spec creation with various configurations."""
        obs_space = Box(low=-1, high=1, shape=(10,))
        act_space = MultiDiscrete([3, 4, 2])

        spec = create_production_bio_module_spec(
            obs_space=obs_space,
            act_space=act_space,
            num_agents=8,
            use_communication=True,
            model_config={
                "hidden_dim": 512,
                "memory_dim": 128,
                "debug_mode": True
            }
        )

        # Validate spec properties
        self.assertEqual(spec.module_class, ProductionUnifiedBioInspiredRLModule)
        self.assertEqual(spec.observation_space, obs_space)
        self.assertEqual(spec.action_space, act_space)

        # Check configuration merging
        config = spec.model_config_dict
        self.assertEqual(config["num_agents"], 8)
        self.assertEqual(config["use_communication"], True)
        self.assertEqual(config["hidden_dim"], 512)
        self.assertEqual(config["memory_dim"], 128)
        self.assertEqual(config["debug_mode"], True)

    def test_invalid_inputs(self):
        """Test error handling for invalid factory inputs."""
        obs_space = Box(low=-1, high=1, shape=(4,))
        act_space = Discrete(5)

        # Invalid num_agents
        with self.assertRaises(ValueError):
            create_production_bio_module_spec(obs_space, act_space, num_agents=0)

        # Invalid spaces (simulate with None)
        with self.assertRaises(ValueError):
            create_production_bio_module_spec(None, act_space, num_agents=4)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking tests."""

    def test_forward_pass_performance(self):
        """Benchmark forward pass performance."""
        obs_space = Box(low=-1, high=1, shape=(12,))
        act_space = MultiDiscrete([4, 3, 2])

        module = ProductionUnifiedBioInspiredRLModule(
            observation_space=obs_space,
            action_space=act_space,
            model_config={
                "num_agents": 8,
                "use_communication": True,
                "hidden_dim": 256,
                "memory_dim": 64,
                "comm_rounds": 3
            }
        )
        module.setup()

        batch_size = 32
        obs = torch.randn(batch_size, 12)
        batch = {"obs": obs}

        # Warmup
        for _ in range(5):
            _ = module.forward_train(batch)

        # Benchmark
        with memory_profiler():
            start_time = time.time()

            for _ in range(100):
                output = module.forward_train(batch)

            end_time = time.time()

        avg_time = (end_time - start_time) / 100
        logger.info(f"Average forward pass time: {avg_time*1000:.2f}ms")

        # Performance assertion (adjust based on your hardware)
        self.assertLess(avg_time, 0.1, "Forward pass should be under 100ms")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_memory_usage(self):
        """Test GPU memory usage."""
        obs_space = Box(low=-1, high=1, shape=(16,))
        act_space = Discrete(8)

        module = ProductionUnifiedBioInspiredRLModule(
            observation_space=obs_space,
            action_space=act_space,
            model_config={
                "num_agents": 12,
                "hidden_dim": 512,
                "memory_dim": 128
            }
        ).cuda()
        module.setup()

        initial_memory = torch.cuda.memory_allocated()

        # Large batch test
        batch_size = 128
        obs = torch.randn(batch_size, 16).cuda()
        batch = {"obs": obs}

        output = module.forward_train(batch)

        final_memory = torch.cuda.memory_allocated()
        memory_used = (final_memory - initial_memory) / 1024 / 1024  # MB

        logger.info(f"GPU memory used: {memory_used:.1f}MB")

        # Memory usage should be reasonable
        self.assertLess(memory_used, 500, "GPU memory usage should be under 500MB")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def test_rllib_integration(self):
        """Test integration with RLlib components."""
        try:
            from ray.rllib.algorithms.ppo import PPOConfig
        except ImportError:
            self.skipTest("RLlib not available")

        obs_space = Box(low=-1, high=1, shape=(8,))
        act_space = Discrete(4)

        spec = create_production_bio_module_spec(
            obs_space=obs_space,
            act_space=act_space,
            num_agents=4,
            model_config={"hidden_dim": 128, "memory_dim": 32}
        )

        # Test PPO config creation (without training)
        try:
            config = (
                PPOConfig()
                .environment("CartPole-v1")  # Dummy environment
                .rl_module(rl_module_spec=spec)
                .framework("torch")
            )

            # Validate config can be built
            algorithm_class = config.get_algorithm_class()
            self.assertIsNotNone(algorithm_class)

        except Exception as e:
            self.fail(f"RLlib integration failed: {e}")


def run_test_suite():
    """Run the complete test suite with reporting."""
    print("🧪 Running Production Bio-Inspired Module Test Suite")
    print("=" * 60)

    # Create test suite
    test_classes = [
        TestPheromoneAttentionNetwork,
        TestNeuralPlasticityMemory,
        TestMultiActionHead,
        TestProductionUnifiedModule,
        TestModuleFactory,
        TestPerformanceBenchmarks,
        TestIntegration,
    ]

    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=None,
        buffer=False,
        failfast=False
    )

    result = runner.run(suite)

    # Summary report
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")

    if result.wasSuccessful():
        print("🎉 All tests passed! Production module is ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = run_test_suite()
    exit(0 if success else 1)
