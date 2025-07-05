# File: marlcomm/debug_utils.py
#!/usr/bin/env python3
"""
Debug and Testing Utilities for Nature-Inspired MARL

This module provides utilities for debugging, testing, and analyzing
the nature-inspired multi-agent reinforcement learning system.
"""

import sys
import torch
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


def test_pytorch_setup():
    """Test PyTorch installation and CUDA availability."""
    print("🔬 Testing PyTorch Setup")
    print("-" * 30)
    
    # Basic PyTorch info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    
    # Test basic tensor operations
    try:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.matmul(x, y)
        print("✅ Basic tensor operations working")
        
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            print("✅ GPU tensor operations working")
        
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return False
    
    return True


def test_rl_module():
    """Test the custom RLModule implementation."""
    print("\n🧠 Testing Custom RLModule")
    print("-" * 30)
    
    try:
        from rl_module import NatureInspiredCommModule
        from gymnasium.spaces import Box, Discrete
        
        # Create test spaces
        obs_space = Box(-1.0, 1.0, (18,))
        act_space = Discrete(5)
        
        # Create module
        module = NatureInspiredCommModule(
            observation_space=obs_space,
            action_space=act_space,
            model_config={"comm_channels": 8}
        )
        module.setup() # Explicitly call setup for testing
        
        print("✅ RLModule creation successful")
        
        # Test forward pass
        batch_size = 4
        obs_batch = torch.randn(batch_size, 18)
        batch = {"obs": obs_batch}
        
        output = module._forward(batch)
        
        print("✅ Forward pass successful")
        print(f"   Output keys: {list(output.keys())}")
        print(f"   Action logits shape: {output['action_dist_inputs'].shape}")
        print(f"   Value predictions shape: {output['vf_preds'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ RLModule test failed: {e}")
        traceback.print_exc()
        return False


def test_environment():
    """Test environment creation and basic functionality."""
    print("\n🌍 Testing Environment")
    print("-" * 30)
    
    try:
        from environment import create_nature_comm_env
        
        # Test environment creation
        env_config = {"debug": False, "num_agents": 3}
        env = create_nature_comm_env(env_config)
        
        print("✅ Environment creation successful")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Active agents: {list(obs.keys())}")
        
        # Test step
        actions = {}
        for agent in obs.keys():
            if hasattr(env.action_space, 'spaces'):
                actions[agent] = env.action_space.spaces[agent].sample()
            else:
                sample = env.action_space.sample()
                actions[agent] = sample[agent] if isinstance(sample, dict) else sample
        
        obs, rewards, dones, truncs, info = env.step(actions)
        print(f"✅ Environment step successful")
        print(f"   Rewards: {rewards}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        traceback.print_exc()
        return False


def test_training_config():
    """Test training configuration creation."""
    print("\n⚙️  Testing Training Configuration")
    print("-" * 30)
    
    try:
        from training_config import TrainingConfigFactory
        from gymnasium.spaces import Box, Discrete
        
        # Create factory
        factory = TrainingConfigFactory()
        
        # Test spaces
        obs_space = {"agent_0": Box(-1.0, 1.0, (18,))}
        act_space = {"agent_0": Discrete(5)}
        
        # Test configuration creation
        config = factory.create_config(obs_space, act_space)
        
        print("✅ Training configuration creation successful")
        print(f"   Framework: {config.framework_str}")
        print(f"   Environment: {config.env}")
        
        # Test config conversion to dict
        config_dict = config.to_dict()
        print("✅ Configuration dictionary conversion successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Training configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_ray_installation():
    """Test Ray and RLlib installation."""
    print("\n☀️ Testing Ray Installation")
    print("-" * 30)
    
    try:
        import ray
        from ray import tune
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec
        
        print(f"✅ Ray version: {ray.__version__}")
        print("✅ Ray RLlib imports successful")
        
        # Test basic PPO config
        config = PPOConfig()
        print("✅ PPO configuration creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Ray installation test failed: {e}")
        return False


def test_dependencies():
    """Test all required dependencies."""
    print("\n📦 Testing Dependencies")
    print("-" * 30)
    
    dependencies = [
        ("torch", "PyTorch"),
        ("gymnasium", "Gymnasium"),
        ("ray", "Ray"),
        ("numpy", "NumPy"),
        ("mpe2", "MPE2"),
        ("supersuit", "SuperSuit"),
    ]
    
    all_good = True
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_good = False
    
    # Optional dependencies
    optional_deps = [
        ("psutil", "PSUtil (for hardware detection)"),
    ]
    
    print("\nOptional dependencies:")
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name}: Not installed (optional)")
    
    return all_good


def run_comprehensive_tests():
    """Run all available tests."""
    print("🧪 Running Comprehensive System Tests")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("PyTorch Setup", test_pytorch_setup),
        ("Ray Installation", test_ray_installation),
        ("Environment", test_environment),
        ("RLModule", test_rl_module),
        ("Training Config", test_training_config),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for training.")
    else:
        print("⚠️  Some tests failed. Please address issues before training.")
    
    return passed == total


def quick_gpu_test():
    """Quick test to verify GPU functionality."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        # Test GPU memory
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        
        # Test memory cleanup
        del x, y, z
        torch.cuda.empty_cache()
        
        print("✅ GPU functionality verified")
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug utilities for Nature-Inspired MARL")
    parser.add_argument("--test", choices=["all", "gpu", "env", "module", "deps"], 
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "all":
        run_comprehensive_tests()
    elif args.test == "gpu":
        quick_gpu_test()
    elif args.test == "env":
        test_environment()
    elif args.test == "module":
        test_rl_module()
    elif args.test == "deps":
        test_dependencies()
